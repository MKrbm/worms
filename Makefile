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
SSOutputDimerOptim = ${local_ham_dir}/SS/array/dimer_optim_J_[${J},1]_M_${M}
SSOutputSinglet = ${local_ham_dir}/SS/array/singlet_J_[${J},1]
SSOutputOriginal = ${local_ham_dir}/SS/array/original
SSResDimerOptim = ${local_ham_dir}/SS/out/dimer_optim_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}_M_${M}
SSResSinglet = ${local_ham_dir}/SS/out/singlet_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}
SSResOriginal = ${local_ham_dir}/SS/out/original_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}


# .PHONY: SSGenLocal
#* define targets
${SSOutputDimerOptim}: 
	@cd ${local_ham_dir}/SS;\
	python make_ss_local.py -l dimer_optim -J ${J} -M ${M} -P ${P} > ${LOGFILE}

${SSOutputSinglet}: 
	@cd ${local_ham_dir}/SS;\
	python make_ss_local.py -l singlet -J ${J} > ${LOGFILE}

${SSOutputOriginal}: 
	@cd ${local_ham_dir}/SS;\
	python make_ss_local.py -l dimer_optim -J ${J} -M ${M} -P ${P} > ${LOGFILE}

${SSResDimerOptim}: ${SSOutputDimerOptim}
	@cd ${execute_dir};\
	../Release/main -m SS2 -ham ${SSOutputDimerOptim} -N ${N} -J1 ${J} -T ${T} \
	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2)) >  ${SSResDimerOptim}

${SSResSinglet}: ArgCheck ${SSOutputSinglet}
	@cd ${execute_dir};\
	../Release/main -m SS2 -ham ${SSOutputSinglet} -N ${N} -J1 ${J} -T ${T} \
	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2)) >  ${SSResSinglet}

${SSResOriginal}: ArgCheck ${SSOutputOriginal}
	@cd ${execute_dir};\
	../Release/main -m SS1 -ham ${SSOutputOriginal} -N ${N} -J1 ${J} \
	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2)) >  ${SSResOriginal}


PHONY:. SSDimerOptim
PHONY:. SSSinglet
PHONY:. SSOriginal

SSDimerOptim: ${SSResDimerOptim}
SSSinglet: ${SSResSinglet}
SSOriginal: ${SSResOriginal}
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