import os
import itertools
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext

class ScaleMultilingualDataset(datasets.IndexBackedDataset):
    
    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'docversion': 'org',
            'doclang': 'ru',
            'd_source_path': '',

            # 'subset': 'all',
            'ranktopk': 1000,
            # 'querysource': 'orgRU-topic',
            'q_source_path': ''
        })
        return result

    def __init__(self, config, vocab, logger):
        super().__init__(config, logger, vocab)
        base_path = os.path.join( util.path_dataset(self), config['docversion'])
        self.index = indices.AnseriniIndex(os.path.join(base_path, 'anserini'), lang=config['doclang'])
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqllite'))

    def path_segment(self):
        result = '{name}@{docversion}_{rankfn}.{ranktopk}_{subset}_{querysource}'.format(**self.config, name=self.name)
        return result

    def _get_index_for_batchsearch(self):
        return self.index
    
    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _lang(self):
        return self.config['doclang']
    
    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt=fmt)
    
    @memoize_method
    def _load_qrels(self, subset, fmt):
        qrels_path = os.path.join(util.path_dataset(self), f'{subset}.qrels')
        return trec.read_qrels_fmt(qrels_path, fmt)

    @memoize_method
    def _load_queries_base(self, subset):
        querysource = self.config['querysource']
        query_path = os.path.join(util.path_dataset(self), f'{subset}.topics')
        return {qid: text for t, qid, text in plaintext.read_tsv(query_path) if t == querysource}
    
    def _load_topics(self, topic_files, source_prefix, qid_prefix=None, encoding=None, xml_prefix=None, format='clef'):
        topics = []
        for topic_file in _join_paths(self.config['q_source_path'], topic_files):
            parser = parse_clef_query_format if format == 'clef' else trec.parse_query_format
            for t, qid, text in parser(open(topic_file, encoding=encoding), xml_prefix=xml_prefix):
                if qid_prefix:
                    qid = qid.replace(qid_prefix, '')
                topics.append((source_prefix+t, qid, text))
        return topics
    
    def _init_collection_iter(self, doc_paths, encoding):
        doc_iter = itertools.chain(*(trec.parse_doc_format(p, encoding) for p in doc_paths))
        doc_iter = self.logger.pbar(doc_iter, desc='documents')
        return doc_iter

    def init(self, force=False):
        raise NotImplementedError

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
