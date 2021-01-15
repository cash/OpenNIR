import os
from itertools import product
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
class ClefRussianDataset(datasets.ClefDataset):
    """
    CLEF 03 04 Russian corpus
    """

    @staticmethod
    def default_config():
        result = datasets.ClefDataset.default_config()
        result.update({
            'subset': 'all',
            'ranktopk': 1000,
            'querysource': 'topic'
        })
        return result

    def __init__(self, config, vocab, logger):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)
        self.index_russian = indices.AnseriniIndex(os.path.join(base_path, 'anserini.ru'), lang=self._lang())
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqllite'))

    def _get_index_for_batchsearch(self):
        return self.index_russian
    
    def _get_index(self, record):
        return self.index_russian

    def _get_docstore(self):
        return self.doc_store

    def _lang(self):
        return 'ru'

    def init(self, force=False):
        # Support Russian queries for now 
        # TODO: make two versions of subset for en and ru
        base_path = util.path_dataset(self)

        all_qrels = trec.read_qrels_dict(open(os.path.join(util.get_working(), 'datasets', 'clef03-04/qrels_ru_2003')))
        all_qrels.update(trec.read_qrels_dict(open(os.path.join(util.get_working(), 'datasets', 'clef03-04/qrels_ru_2004'))))
        
        all_topics = []
        for topic_file in _join_paths(['clef03-04/Top-ru03.txt', 'clef03-04/Top-ru04.txt']):
            for t, qid, text in parse_clef_query_format(open(topic_file, encoding='UTF-8-SIG'), xml_prefix='RU-'):
                qid = qid.replace('C', '')
                all_topics.append((t, qid, text))

        for fold in FOLDS:
            fold_qrels_file = os.path.join(base_path, f'{fold}.qrels')
            if (force or not os.path.exists(fold_qrels_file)):
                fold_qrels = {qid: dids for qid, dids in all_qrels.items() if qid in FOLDS[fold]}
                trec.write_qrels_dict(fold_qrels_file, fold_qrels)
            
            fold_topic_file = os.path.join(base_path, f'{fold}.topics')
            if (force or not os.path.exists(fold_topic_file)):
                plaintext.write_tsv(fold_topic_file, [ r for r in all_topics if r[1] in FOLDS[fold] ])

        # # For reference
        # for lang, year in product(['en', 'ru'], ['03', '04']):
        #     self._init_topics(
        #         subset=f'{lang}{year}',
        #         topic_files=[f'clef03-04/Top-{lang}{year}.txt'],
        #         heldout_topics=HELDOUT_VALD_03 if year == '03' else [],
        #         qid_prefix='C',
        #         encoding="ISO-8859-1" if lang == 'en' else 'UTF-8-SIG',
        #         xml_prefix=f'{lang}-'.upper(),
        #         force=force)

        #     self._init_qrels(
        #         subset=f'{lang}{year}',
        #         heldout_topics=HELDOUT_VALD_03 if year == '03' else [],
        #         qrels_files=[f'clef03-04/qrels_ru_20{year}'],
        #         force=force)

        self._init_indices_parallel(
            indices=[self.index_russian, self.doc_store],
            doc_iter=self._init_collection_iter(
                doc_paths=['clef03-04/rudocs'],
                encoding="utf-8"),
            force=force)
