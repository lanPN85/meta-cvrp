download-data:
	$(MAKE) -C data/

# CSV files containing table data
report/data/cost_augA_1.csv:
	./gen_csv_1.py \
		-o $@ \
		-i1 results/CW_augerat_A/ \
		-i2 results/LS_augerat_A/ \
		-i3 results/CBC_augerat_A/

report/data/cost_augB_1.csv:
	./gen_csv_1.py \
		-o $@ \
		-i1 results/CW_augerat_B/ \
		-i2 results/LS_augerat_B/ \
		-i3 results/CBC_augerat_B/

clean-report-data:
	rm report/data/*

build-report-data:
	$(MAKE) \
		report/data/cost_augA_1.csv \
		report/data/cost_augB_1.csv

rebuild-report-data: clean-data
	$(MAKE) build-data

report: build-data
	$(MAKE) -C report/

.PHONY: report rebuild-report-data build-report-data clean-report-data download-data
