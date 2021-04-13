srcL:=$(wildcard /mnt/stage/douwei/Simulation/1t_root/point_axis_10/1t_+*.root)
dstL:=$(srcL:%.root=%_Q.h5)

.PHONY: all
all: $(dstL)

%_Q.h5: %.root
	nice -n 19 python3 ConvertTruthMerge.py $^ $@

.DELETE_ON_ERROR:
.SECONDARY:
