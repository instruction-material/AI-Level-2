# Freeze an evaluation contract before modeling

The supplied 240 fictional ticket summaries contain question=120,bug=80 and request=40. Use summary as prediction input,label as target,and template_family as the split group. ticket_id is for tracing. Repeated template families stay together. This templated case checks evaluation procedure rather than performance on real correspondence.

Implement build_contract in the starter using the declared return fields in the supplied main program and contract below. Group rows by template_family; every family has one label. Sort class labels and each class's family names,then shuffle each class's family list using one local Random(seed)generator. Reserve max(1,round(0.2*family_count))whole families for test and the same count for validation; retain the rest for training.

Return seed,prediction_input,target,split_group,trace_identifier,class_counts,split_rows,split_ids,split_families and split_class_counts. split_ids and split_families are sorted lists for each partition; class counts cover all three labels in every partition.

Also return the evaluation plan: baseline={"class":"DummyClassifier","strategy":"most_frequent"},primary_metric="macro_f1",secondary_metric="balanced_accuracy",vectorization fitted only on training summaries inside its candidate pipeline,two grouped training folds,error_slices for summary length and error codes,a simplicity rule requiring at least 0.02 validation macro-F1 improvement,and a test_gate requiring candidate/preprocessing/parameter decisions to be frozen before final test prediction. The source contains no fitted model.

With seed 42, the expected split sizes are 132 training, 54 validation and 54 test rows. Their class counts are respectively question72/bug40/request20,question24/bug20/request10 and question24/bug20/request10. Whole-group rounding makes these row fractions differ from exact 60/20/20. Class coverage and disjoint template families are required; report the actual row fractions.

Use Python 3.10 or later with its standard library. Run python3 main.py,optionally adding --output contract.json. Each role includes its CSV. Restart and verify the same IDs,counts and source checksum. Compare another seed as optional transfer,then freeze one plan before training any candidate.

Role: starter.
