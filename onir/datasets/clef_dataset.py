import os
import gzip
import itertools
from pytools import memoize_method
from onir import util, datasets
from onir.interfaces import trec, plaintext


class ClefDataset(datasets.IndexBackedDataset):
    """
    Abstract class for CLEF03/04 dataset
    """
    DUA = """Will use CLEF03 04 data locally from `source_path`"""


    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': '', #TODO supporting 03/04 en/ru topics
            'ranktopk': 1000,
            'querysource': 'topic',
            'source_path': ''
        })
        return result

    def path_segment(self):
        result = '{name}_{rankfn}.{ranktopk}_{subset}'.format(**self.config, name=self.name)
        if self.config['querysource'] != 'topic':
            result += '_{querysource}'.format(**self.config)
        return result

    def _lang(self):
        raise NotImplementedError()

    @memoize_method
    def _load_queries_base(self, subset):
        querysource = self.config['querysource']
        query_path = os.path.join(util.path_dataset(self), f'{subset}.topics')
        return {qid: text for t, qid, text in plaintext.read_tsv(query_path) if t == querysource}

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt=fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        qrels_path = os.path.join(util.path_dataset(self), f'{subset}.qrels')
        return trec.read_qrels_fmt(qrels_path, fmt)

    def _init_topics(self, subset, topic_files, heldout_topics=None, qid_prefix=None, encoding=None, xml_prefix=None, force=False, expected_md5=None):
        topicf = os.path.join(util.path_dataset(self), f'{subset}.topics')
        topicf_heldout = os.path.join(util.path_dataset(self), f'{subset}-heldout.topics')
        heldout_topics = heldout_topics or []
        if (force or not os.path.exists(topicf)) and self._confirm_dua():
            topics, topics_heldout = [], []
            for topic_file in _join_paths(self.config['source_path'], topic_files):
                opener = gzip.open if topic_file.endswith('.gz') else open
                for t, qid, text in parse_clef_query_format(opener(topic_file, encoding=encoding), xml_prefix):
                    if qid_prefix is not None:
                        qid = qid.replace(qid_prefix, '')
                    if qid in heldout_topics:
                        topics_heldout.append((t, qid, text))
                    else:
                        topics.append((t, qid, text))
            plaintext.write_tsv(topicf, topics)
            if len(topics_heldout) > 0:
                plaintext.write_tsv(topicf_heldout, topics_heldout)

    def _init_qrels(self, subset, qrels_files, heldout_topics=None, force=False, expected_md5=None):
        qrelsf = os.path.join(util.path_dataset(self), f'{subset}.qrels')
        qrelsf_heldout = os.path.join(util.path_dataset(self), f'{subset}-heldout.qrels')
        heldout_topics = heldout_topics or []
        if (force or not os.path.exists(qrelsf)) and self._confirm_dua():
            qrels = itertools.chain(*(trec.read_qrels(open(f)) for f in _join_paths(self.config['source_path'], qrels_files) ))
            trec.write_qrels(qrelsf, (q for q in qrels if q[0] not in heldout_topics))
            if len(heldout_topics) > 0:
                qrels = itertools.chain(*(trec.read_qrels(open(f)) for f in _join_paths(self.config['source_path'], qrels_files) ))
                trec.write_qrels(qrelsf_heldout, (q for q in qrels if q[0] in heldout_topics))


    def _init_collection_iter(self, doc_paths, encoding):
        doc_paths = _join_paths(self.config['source_path'], doc_paths)
        doc_iter = itertools.chain(*(trec.parse_doc_format(p, encoding) for p in doc_paths))
        doc_iter = self.logger.pbar(doc_iter, desc='documents')
        return doc_iter

def _join_paths(source, paths):
    return (os.path.join(source, p) for p in paths)

def parse_clef_query_format(file, xml_prefix=None):
    if xml_prefix is None:
        xml_prefix = ''
    num, title, desc, narr, reading = None, None, None, None, None
    for line in file:
        if line.startswith('**'):
            continue # translation comment in older formats (e.g., TREC 3 Spanish track)
        elif line.startswith('</top>'):
            if title is not None:
                yield 'topic', num, title.replace('\t', ' ').strip()
            if desc is not None:
                yield 'desc', num, desc.replace('\t', ' ').strip()
            if narr is not None:
                yield 'narr', num, narr.replace('\t', ' ').strip()
            num, title, desc, narr, reading = None, None, None, None, None
        elif line.startswith('<num>'):
            num = line[len('<num>'):].replace('Number:', '').replace('</num>', '').strip()
            reading = None
        elif line.startswith(f'<{xml_prefix}title>'):
            title = line[len(f'<{xml_prefix}title>'):-len(f'</{xml_prefix}title>')-1].strip()
        elif line.startswith(f'<{xml_prefix}desc>'):
            desc = line[len(f'<{xml_prefix}desc>'):-len(f'</{xml_prefix}desc>')-1].strip()
        elif line.startswith(f'<{xml_prefix}narr>'):
            narr = line[len(f'<{xml_prefix}narr>'):-len(f'</{xml_prefix}narr>')-1].strip()
