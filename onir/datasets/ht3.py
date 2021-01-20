import os
import itertools
import gzip
import json
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext


class HT3Dataset(datasets.ScaleMultilingualDataset):
    """Abstract class for HT3 CC dataset. 
    - Support different versions of documents and qrels
    """

    @staticmethod
    def default_config():
        result = datasets.ScaleMultilingualDataset.default_config()
        result.update({
            'docversion': '',
            'qrelsversion': '',
            'querysource': '',
            'subset': 'all',
            'ranktopk': 1000,
        })
        return result
    
    def path_segment(self):
        result = '{name}@{docversion}^{qrelsversion}_{rankfn}.{ranktopk}_{subset}_{querysource}'\
                 .format(**self.config, name=self.name)
        return result
    
    def qrels(self, fmt='dict'):
        # to support different version of qrels
        return self._load_qrels(self.config['subset'], self.config['qrelsversion'], fmt=fmt)
    
    @memoize_method
    def _load_qrels(self, subset, version, fmt):
        # to support different version of qrels
        qrels_path = os.path.join(util.path_dataset(self), f'{subset}.{version}.qrels')
        return trec.read_qrels_fmt(qrels_path, fmt)
    
    def _load_topics(self, topic_files, source_prefix, encoding=None):
        topics = []
        for topic_file in (os.path.join(self.config['q_source_path'], f) for f in topic_files):
            for t, qid, text in parse_cc_topics(open(topic_file, encoding=encoding)):
                topics.append((source_prefix+t, qid, text))
        return topics
    
    def _init_qrels(self, folds, version, force):
        base_path = util.path_dataset(self)

        for fold in folds:
            fold_qrels_file = os.path.join(base_path, f'{fold}.{version}.qrels')
            if (force or not os.path.exists(fold_qrels_file)) and self._confirm_dua():
                all_qrels = self._load_all_qrels()
                fold_qrels = {qid: dids for qid, dids in all_qrels.items() if qid in folds[fold]}
                trec.write_qrels_dict(fold_qrels_file, fold_qrels)

    def _init_topics(self, folds, force):
        base_path = util.path_dataset(self)

        for fold in folds:
            fold_topic_file = os.path.join(base_path, f'{fold}.topics')
            if (force or not os.path.exists(fold_topic_file)) and self._confirm_dua():
                all_topics = self._load_all_topics()
                plaintext.write_tsv(fold_topic_file, [ r for r in all_topics if r[1] in folds[fold] ])

    def _init_collection_iter(self, doc_paths, encoding):
        doc_iter = itertools.chain(*(parse_cc_docs(p, encoding) for p in doc_paths))
        doc_iter = self.logger.pbar(doc_iter, desc='documents')
        return doc_iter
    
def parse_cc_docs(path, encoding='utf8'):
    if os.path.isdir(path): 
        for file in os.listdir(path):
            yield from _parse_doc_format_files(os.path.join(path, file))
    else:
        opener = gzip.open if path.endswith('.gz') else open
        for l in opener(path, encoding=encoding):
            #TODO: split it to different parser function if different file formats will be used
            d = json.loads(l)
            yield indices.RawDoc( d['id'], text=d['text'], title=d['title'])

def parse_cc_topics(file):
    for l in file:
        d = json.loads(l)
        yield from [ ('title', d['topic_id'], d['topic_name']),
                     ('desc', d['topic_id'], d['topic_description']) ]