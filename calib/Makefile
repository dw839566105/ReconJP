.PHONY: all

mode:=time
basis:=Legendre
alg:=sms
order:=10
qt:=0.5

path:=/mnt/stage/douwei/Simulation/1t_root/
path2:=point_10_photon_2MeV
srcL:=$(wildcard $(path)$(path2)/*total.h5)
dstL:=$(srcL:$(path)$(path2)/%_total.h5=coeff_$(path2)/$(mode)_$(order)_%.h5)

srcL:=$(wildcard $(path)$(path2)/*total.h5)
dstL:=$(srcL:$(path)$(path2)/%_total.h5=coeff_$(path2)/$(mode)_$(order)_$(qt)_%.h5)

all: $(dstL)

#coeff_$(path2)/$(mode)_$(order)_%.h5: $(path)$(path2)/%_total.h5
coeff_$(path2)/$(mode)_$(order)_$(qt)_%.h5: $(path)$(path2)/%_total.h5
	mkdir -p coeff_$(path2)
	python3 main.py -f $^ -o $@ --order $(order) -v 1 --alg $(alg) --basis $(basis) --mode $(mode) --qt $(qt) > $@.log

.DELETE_ON_ERROR:

.SECONDARY:
