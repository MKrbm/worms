# var = $(shell cat in.txt)

# out : out.txt

# out.txt: in.txt
# 	cat in.txt > out.txt

# max = 100
# OUT_FILENAME = out_max_${max}.txt
# # .PHONY: out2
# # out2 := ${OUT_FILENAME}
# ${OUT_FILENAME}: out.txt in.txt
# 	echo ${var}
# 	echo "$(var)+a" > ${OUT_FILENAME}


# test.txt:
# 	python  test_multiprocess.py > test.txt

# test2.txt: ${OUT_FILENAME}
# 	echo ${max}
# 	cat ${OUT_FILENAME} > test2.txt



# make_local := ./make_local
execute_dir := /home/user/project/execute
python_dir := /home/user/project/python/reduce_nsp
local_ham_dir := ${python_dir}/make_local

LOGPATH = /home/user/project/python/reduce_nsp/make_local/logs
LOGFILE = $(LOGPATH)/$(shell date --iso=seconds)

#* for shastry-surtherland
M = 120
N = 100000
P = 12
L = 4
T = 1.0
ArgCheck:
ifeq ($(L),)
	$(error Lattice size is not specified.)
else ifeq ($(J),)
	$(error No coupling constant is specified.)
endif
# order : L, J, T, N, M
SSOutputDimerOptim = J_${J}.txt
SSOutputSinglet = ${local_ham_dir}/SS/array/singlet_J_[${J},1]
SSOutputOriginal = ${local_ham_dir}/SS/array/original
SSResDimerOptim = ${local_ham_dir}/SS/out/dimer_optim_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}_M_${M}
SSResSinglet = ${local_ham_dir}/SS/out/singlet_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}
SSResOriginal = ${local_ham_dir}/SS/out/original_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}

PHONY:. SSDimerOptim

# .PHONY: SSGenLocal
#* define targets
${SSOutputDimerOptim}: 
	echo "${M}_${T}_${J}" > ${SSOutputDimerOptim}

${SSResDimerOptim}: ${SSOutputDimerOptim}
	cat ${SSOutputDimerOptim} > ${SSResDimerOptim}

SSDimerOptim : ${SSResDimerOptim}
# ifeq ($(lat), dimer_optim)
# 	@cd ${execute_dir};\
# 	../Release/main -m SS2 -ham ${SSOutput} -N ${N} -J1 ${J} \
# 	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2)) >>  ${SSres}
# else ifeq ($(lat), dimer_basis)
# 	@cd ${execute_dir};\
# 	../Release/main -m SS2 -ham ${SSOutput} -N ${N} -J1 ${J} \
# 	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2)) >>  ${SSres}
# else
# 	echo "Hi";\
# 	cd ${execute_dir};\
# 	../Release/main -m SS1 -ham ${SSOutput} -N ${N} -J1 ${J} \
# 	-L1 $(L) -L2 $(L) >>  ${SSres}
# endif


# cleanSS:
# 	rm -rf ${local_ham_dir}/SS/array/*