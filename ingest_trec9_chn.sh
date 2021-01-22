scripts/init_dataset.sh dataset=trec9_chn vocab=trivial \
                        dataset.d_source_path=/home/hltcoe/eyang/data/datasets/trec9/docs \
                        dataset.docversion=org dataset.doclang=zh \
                        dataset.q_source_path=/home/hltcoe/eyang/data/datasets/trec9;

scripts/init_dataset.sh dataset=trec9_chn vocab=trivial \
                        dataset.d_source_path=/home/hltcoe/eyang/data/datasets/trec9/docs-in-en/trec9.20210115-scale18rm1prelim.sgml \
                        dataset.docversion=mtzhen20210115.scale18rm1prelim dataset.doclang=en \
                        dataset.q_source_path=/home/hltcoe/eyang/data/datasets/trec9;


for mtmodel in scale18rm1-ersatz scale18rm1-spacy scale18ts1-ersatz scale18ts1-spacy; do
    yes | scripts/init_dataset.sh dataset=trec9_chn vocab=trivial \
                            dataset.d_source_path=/home/hltcoe/eyang/data/datasets/trec9/docs-in-en/zh_trec9.20210121-$mtmodel.sgml \
                            dataset.docversion=mtzhen20210121.$mtmodel dataset.doclang=en \
                            dataset.q_source_path=/home/hltcoe/eyang/data/datasets/trec9;
done