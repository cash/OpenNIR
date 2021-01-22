for mtmodel in scale18rm1-ersatz scale18rm1-spacy scale18ts1-ersatz scale18ts1-spacy; do
    scripts/pipeline.sh config/trivial config/trec9/orgEN-topic\
                        train_ds.docversion=mtzhen20210121.$mtmodel \
                        valid_ds.docversion=mtzhen20210121.$mtmodel \
                        test_ds.docversion=mtzhen20210121.$mtmodel ;
done

for mtmodel in scale18rm1-ersatz scale18rm1-spacy scale18ts1-ersatz scale18ts1-spacy; do
    scripts/pipeline.sh config/trivial config/trec9/orgEN-desc\
                        train_ds.docversion=mtzhen20210121.$mtmodel \
                        valid_ds.docversion=mtzhen20210121.$mtmodel \
                        test_ds.docversion=mtzhen20210121.$mtmodel ;
done