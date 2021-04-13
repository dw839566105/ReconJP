.PHONY: all

data:=point_10_photon_2MeV
prog:=main
path:=/mnt/stage/douwei/Simulation/1t_root/$(data)/
save:=result_1t_$(data)

sim:=$(wildcard $(path)1t_+0.??0.root)
dst:=$(sim:$(path)%.root=$(save)/%.h5)
all: $(dst)

$(save)/%.h5: $(path)%.root
	mkdir -p $(save)
	python3 $(prog).py -f $^ -o $@ > $@.log

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
