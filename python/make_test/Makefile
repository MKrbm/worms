# make_local := ./make_local
execute_dir := /home/user/project/execute
python_dir := /home/user/project/python/reduce_nsp
local_ham_dir := ${python_dir}/make_local

#* for log files
LOGPATH = /home/user/project/python/reduce_nsp/make_local/logs
LOGFILE = $(LOGPATH)/$(shell date --iso=seconds)



#* for shastry-surtherland
M = 120
N = 100000
P = 12
L = 4
T = 1.0

PHONY:. Check
Check:
ifeq ($(L),)
	$(error Lattice size is not specified.)
else ifeq ($(J),)
	$(error No coupling constant is specified.)
endif


A := a.txt
B := b.txt
C := c.txt

$(A) : $(B)
	echo $(ARGS)
	$(MAKE) Check $(ARGS)
	echo "$(shell cat ${B})abc" > ${A}

$(B) : $(C)
	echo  "$(shell cat ${C})" > ${B}