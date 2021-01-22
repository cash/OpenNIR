import os
import itertools
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext
from onir.datasets.scale_multilingual_dataset import _join_paths, parse_clef_query_format

FOLDS = {
    'f1': {'69', '68', '72', '59', '62'},
    'f2': {'55', '76', '70', '63', '66'},
    'f3': {'56', '57', '61', '78', '77'},
    'f4': {'79', '71', '67', '65', '60'},
    'f5': {'64', '75', '73', '74', '58'}
}
_ALL = set.union(*FOLDS.values())
_FOLD_IDS = list(sorted(FOLDS.keys()))
for i in range(len(FOLDS)):
    FOLDS['tr' + _FOLD_IDS[i]] = _ALL - FOLDS[_FOLD_IDS[i]] - FOLDS[_FOLD_IDS[i-1]]
    FOLDS['va' + _FOLD_IDS[i]] = FOLDS[_FOLD_IDS[i-1]]
FOLDS['all'] = _ALL


@datasets.register('trec9_chn')
class Trec9ChineseDataset(datasets.ScaleMultilingualDataset):
    """
    TREC 9 Chinese dataset
    """
    DUA = """Will use TREC 9 data locally from and `d_source_path`, `q_source_path`"""

    @staticmethod
    def default_config():
        result = datasets.ScaleMultilingualDataset.default_config()
        result.update({
            'docversion': 'org',
            'doclang': 'zh',

            'subset': 'all',
            'ranktopk': 1000,
            'querysource': 'orgZH-topic'
        })
        return result

    @memoize_method
    def _load_all_topics(self):
        return [
            *self._load_topics(topic_files=['topics/xling9-topics.txt'], source_prefix='orgEN-', 
                               qid_prefix='CH', format='trec'),
            *self._load_topics(topic_files=['topics/xling9-topics.chinese'], source_prefix='orgZH-', 
                               qid_prefix='CH', encoding='Big5', format='trec'),
            *self._load_topics(topic_files=['topics/xling9-topics.goog-enzh.txt'], source_prefix='googENZH-', 
                               qid_prefix='CH', encoding='utf8', format='trec'),
            #TODO: add interface so that the output from MT team doesn't need to be massaged
            # *self._load_topics(topic_files=['topics/trec9topics.20210121-m2m100-1.2b.sgml'], source_prefix='mt0121-m2m100-1.2b-', 
            #                    qid_prefix='CH', encoding='UTF-8', xml_prefix=''),
            # *self._load_topics(topic_files=['topics/trec9topics.20210121-m2m100-418m.sgml'], source_prefix='mt0121-m2m100-418m-', 
            #                    qid_prefix='CH', encoding='UTF-8', xml_prefix=''),
        ]

    @memoize_method
    def _load_all_qrels(self):
        all_qrels = trec.read_qrels_dict(open(os.path.join(self.config['q_source_path'], 'qrels/xling9_qrels')))
        return all_qrels
    
    def init(self, force=False):
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
