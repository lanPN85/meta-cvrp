
# CSV files containing table data
report/data/cost_augA_cwls.csv:
	./gen_csv_cwls.py \
		-o report/data/cost_augA_cwls.csv \
		-i1 results/CW_augerat_A/ \
		-i2 results/LS_augerat_A/

report/data/cost_augB_cwls.csv:
	./gen_csv_cwls.py \
		-o report/data/cost_augB_cwls.csv \
		-i1 results/CW_augerat_B/ \
		-i2 results/LS_augerat_B/

clean-data:
	rm report/data/*

build-data:
	$(MAKE) \
		report/data/cost_augA_cwls.csv \
		report/data/cost_augB_cwls.csv

rebuild-data: clean-data
	$(MAKE) build-data

report: build-data
	$(MAKE) -C report/

.PHONY: report rebuild-data build-data clean-data
