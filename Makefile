download-data:
	$(MAKE) -C data/

# Tables for report
report/tables/compare_cmt_others.tex: gen_compare_others.py
	./gen_compare_others.py -d cmt \
		-l "tab:compare-others-cmt" \
		-c "Best total cost for SimpleTabu, ALNS and GRELS on the Christofides 1979 dataset" \
		> $@

report/tables/compare_cmt_abl.tex: gen_compare_abl.py
	./gen_compare_abl.py -d cmt \
		-l "tab:compare-abl-cmt" \
		-c "Best total cost and standard deviation over 5 runs for variations of SimpleTabu on the Christofides 1979 dataset" \
		> $@

clean-tables:
	rm report/tables/*

build-tables:
	$(MAKE) \
		report/tables/compare_cmt_others.tex \
		report/tables/compare_cmt_abl.tex

# Images for report
report/images/converge_1.jpg:
	./draw_converge_chart.py \
		-i results/LS_cmt/version_0/summary.yml \
		-n CMT07 -r 0 -o $@

report/images/converge_2.jpg:
	./draw_converge_chart.py \
		-i results/LS_cmt/version_1/summary.yml \
		-n CMT12 -r 0 -o $@

build-images:
	$(MAKE) \
		report/images/converge_1.jpg \
		report/images/converge_2.jpg

report: build-tables build-images
	$(MAKE) -C report/

.PHONY: report download-data
