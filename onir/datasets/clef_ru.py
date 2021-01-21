import os
import itertools
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext
from onir.datasets.scale_multilingual_dataset import _join_paths, parse_clef_query_format

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
class ClefRussianDataset(datasets.ScaleMultilingualDataset):
    """
    CLEF 03 04 Russian corpus
    """
    DUA = """Will use CLEF03 04 data locally from and `d_source_path`, `q_source_path`"""

    @staticmethod
    def default_config():
        result = datasets.ScaleMultilingualDataset.default_config()
        result.update({
            'docversion': 'org',
            'doclang': 'ru',

            'subset': 'all',
            'ranktopk': 1000,
            'querysource': 'orgRU-topic',
        })
        return result

    @memoize_method
    def _load_all_topics(self):
        return [
            *self._load_topics(topic_files=['Top-ru03.txt', 'Top-ru04.txt'], source_prefix='orgRU-', 
                               qid_prefix='C', encoding='UTF-8-SIG', xml_prefix='RU-'),
            *self._load_topics(topic_files=['Top-en03.txt', 'Top-en04.txt'], source_prefix='orgEN-', 
                               qid_prefix='C', encoding='ISO-8859-1', xml_prefix='EN-'),
            *self._load_topics(topic_files=['Top-en0304-gt-ru-filtered.txt'], source_prefix='googENRU-', 
                               qid_prefix='C', encoding='UTF-8', xml_prefix='RU-'),
            #TODO: add interface so that the output from MT team doesn't need to be massaged
            *self._load_topics(topic_files=['cleftopics.20210121-m2m100-1.2b.sgml'], source_prefix='mt0121-m2m100-1.2b-', 
                               qid_prefix='C', encoding='UTF-8', xml_prefix='RU-'),
            *self._load_topics(topic_files=['cleftopics.20210121-m2m100-418m.sgml'], source_prefix='mt0121-m2m100-418m-', 
                               qid_prefix='C', encoding='UTF-8', xml_prefix='RU-'),
        ]

    @memoize_method
    def _load_all_qrels(self):
        all_qrels = trec.read_qrels_dict(open(os.path.join(self.config['q_source_path'], 'qrels_ru_2003')))
        all_qrels.update(trec.read_qrels_dict(open(os.path.join(self.config['q_source_path'], 'qrels_ru_2004'))))
        return all_qrels
    
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
