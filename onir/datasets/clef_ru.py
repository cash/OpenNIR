import os
import itertools
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext
from onir.datasets.clef_dataset import _join_paths, parse_clef_query_format

FOLDS = {
    'f1': {'192', '220', '157', '214', '164', '153', '211', '227', '216', '169', '215', '242', '250'},
    'f2': {'181', '212', '149', '202', '245', '193', '209', '224', '177', '163', '207', '244', '176'},
    'f3': {'241', '143', '179', '151', '237', '201', '234', '198', '221', '155', '203', '231'},
    'f4': {'213', '232', '197', '172', '239', '210', '168', '183', '225', '154', '187', '228'},
    'f5': {'230', '199', '200', '148', '235', '233', '178', '180', '218', '238', '147', '226'}
}
_ALL = set.union(*FOLDS.values())
_FOLD_IDS = list(sorted(FOLDS.keys()))
for i in range(len(FOLDS)):
    FOLDS['tr' + _FOLD_IDS[i]] = _ALL - FOLDS[_FOLD_IDS[i]] - FOLDS[_FOLD_IDS[i-1]]
    FOLDS['va' + _FOLD_IDS[i]] = FOLDS[_FOLD_IDS[i-1]]
FOLDS['all'] = _ALL


@datasets.register('clef0304_ru')
class ClefRussianDataset(datasets.IndexBackedDataset):
    """
    CLEF 03 04 Russian corpus
    """
    DUA = """Will use CLEF03 04 data locally from and `d_source_path`, `q_source_path`"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'docversion': 'org',
            'doclang': 'ru',
            'd_source_path': '',

            'subset': 'all',
            'ranktopk': 1000,
            'querysource': 'orgRU-topic',
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
    
    def _load_topics(self, topic_files, source_prefix, qid_prefix=None, encoding=None, xml_prefix=None):
        topics = []
        for topic_file in _join_paths(self.config['q_source_path'], topic_files):
            for t, qid, text in parse_clef_query_format(open(topic_file, encoding=encoding), xml_prefix=xml_prefix):
                if qid_prefix:
                    qid = qid.replace(qid_prefix, '')
                topics.append((source_prefix+t, qid, text))
        return topics
    
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

    @memoize_method
    def _load_all_topics(self):
        return [
            *self._load_topics(topic_files=['Top-ru03.txt', 'Top-ru04.txt'], source_prefix='orgRU-', 
                               qid_prefix='C', encoding='UTF-8-SIG', xml_prefix='RU-'),
            *self._load_topics(topic_files=['Top-en03.txt', 'Top-en04.txt'], source_prefix='orgEN-', 
                               qid_prefix='C', encoding='ISO-8859-1', xml_prefix='EN-'),
            *self._load_topics(topic_files=['Top-en0304-gt-ru-filtered.txt'], source_prefix='googENRU-', 
                               qid_prefix='C', encoding='UTF-8', xml_prefix='RU-')
        ]

    @memoize_method
    def _load_all_qrels(self):
        all_qrels = trec.read_qrels_dict(open(os.path.join(self.config['q_source_path'], 'qrels_ru_2003')))
        all_qrels.update(trec.read_qrels_dict(open(os.path.join(self.config['q_source_path'], 'qrels_ru_2004'))))
        return all_qrels
    
    def _init_collection_iter(self, doc_paths, encoding):
        doc_iter = itertools.chain(*(trec.parse_doc_format(p, encoding) for p in doc_paths))
        doc_iter = self.logger.pbar(doc_iter, desc='documents')
        return doc_iter

    def init(self, force=False):
        # Support Russian queries for now 
        base_path = util.path_dataset(self)

        for fold in FOLDS:
            fold_qrels_file = os.path.join(base_path, f'{fold}.qrels')
            if (force or not os.path.exists(fold_qrels_file)) and self._confirm_dua():
                all_qrels = self._load_all_qrels()
                fold_qrels = {qid: dids for qid, dids in all_qrels.items() if qid in FOLDS[fold]}
                trec.write_qrels_dict(fold_qrels_file, fold_qrels)

            fold_topic_file = os.path.join(base_path, f'{fold}.topics')
            if (force or not os.path.exists(fold_topic_file)) and self._confirm_dua():
                all_topics = self._load_all_topics()
                plaintext.write_tsv(fold_topic_file, [ r for r in all_topics if r[1] in FOLDS[fold] ])

        self._init_indices_parallel(
            indices=[self.index, self.doc_store],
            doc_iter=self._init_collection_iter(
                doc_paths=[ self.config['d_source_path'] ],
                encoding="utf-8"),
            force=force)
