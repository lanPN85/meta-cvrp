download-data:
	$(MAKE) -C data/

clean-tables:
	rm report/tables/*

build-tables:
	$(MAKE) \
		report/data/cost_augA_1.csv \
		report/data/cost_augB_1.csv

rebuild-tables: clean-tables
	$(MAKE) build-data

report: build-tables
	$(MAKE) -C report/

.PHONY: report download-data
