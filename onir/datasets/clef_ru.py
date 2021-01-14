import os
from itertools import product
from onir import util, datasets, indices

HELDOUT_VALD_03 = ['150', '199', '152', '184', '176', '164']


@datasets.register('clef0304_ru')
class ClefRussianDataset(datasets.ClefDataset):
    """
    CLEF 03 04 Russian corpus
    """

    @staticmethod
    def default_config():
        result = datasets.ClefDataset.default_config()
        result.update({
            'subset': 'ru03',
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

    def _get_docstore(self):
        return self.doc_store

    def _lang(self):
        return 'ru'

    def init(self, force=False):
        # Support both English and Russian queries
        # TODO: how to relate Russian queries to English docs(unclear for now)

        for lang, year in product(['en', 'ru'], ['03', '04']):
            self._init_topics(
                subset=f'{lang}{year}',
                topic_files=[f'clef03-04/Top-{lang}{year}.txt'],
                heldout_topics=HELDOUT_VALD_03 if year == '03' else [],
                qid_prefix='C',
                encoding="ISO-8859-1" if lang == 'en' else 'UTF-8-SIG',
                xml_prefix=f'{lang}-'.upper(),
                force=force)

            self._init_qrels(
                subset=f'{lang}{year}',
                heldout_topics=HELDOUT_VALD_03 if year == '03' else [],
                qrels_files=[f'clef03-04/qrels_ru_20{year}'],
                force=force)

        self._init_indices_parallel(
            indices=[self.index_russian, self.doc_store],
            doc_iter=self._init_collection_iter(
                doc_paths=['clef03-04/rudocs'],
                encoding="utf-8"),
            force=force)
