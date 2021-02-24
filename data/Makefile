default: golden-1998 christofides-1979

golden-1998:
	mkdir $@
	curl http://www.vrp-rep.org/datasets/download/golden-et-al-1998-set-1.zip > $@/data.zip
	cd $@ && unzip data.zip
	rm $@/data.zip

christofides-1979:
	mkdir $@
	curl http://www.vrp-rep.org/datasets/download/christofides-et-al-1979-cmt.zip > $@/data.zip
	cd $@ && unzip data.zip
	rm $@/data.zip

.PHONY: default
