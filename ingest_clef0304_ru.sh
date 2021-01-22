scripts/init_dataset.sh dataset=clef0304_ru vocab=trivial \
                        dataset.d_source_path=/home/hltcoe/eyang/data/datasets/clef03-04/rudocs \
                        dataset.q_source_path=/home/hltcoe/eyang/data/datasets/clef03-04;

scripts/init_dataset.sh dataset=clef0304_ru vocab=trivial \
                        dataset.d_source_path=/home/hltcoe/eyang/data/datasets/clef03-04/rudocs-in-en \
                        dataset.docversion=mtruen20210115 dataset.doclang=en \
                        dataset.q_source_path=/home/hltcoe/eyang/data/datasets/clef03-04

# source: /exp/scale21/translation/done/ru-en/clef04.20210115-m2m100-small.sgml
scripts/init_dataset.sh dataset=clef0304_ru vocab=trivial \
                        dataset.d_source_path=/home/hltcoe/eyang/data/datasets/clef03-04/rudocs-in-en/clef04.20210115-m2m100-small.sgml \
                        dataset.docversion=mtruen20210115.m2m100-small dataset.doclang=en \
                        dataset.q_source_path=/home/hltcoe/eyang/data/datasets/clef03-04

