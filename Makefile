# make_local := ./make_local
execute_dir := /home/user/project/execute
python_dir := /home/user/project/python
local_ham_dir := ${python_dir}/make_local

#* for log files
LOGPATH = /home/user/project/python/make_local/logs
LOGFILE = $(LOGPATH)/$(shell date --iso=seconds)



#* for shastry-surtherland
M = 120
N = 100000
P = 12
L = 4
T = 1.0

PHONY:. ArgCheck
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
SSResDimerOptim = ${local_ham_dir}/out/SS/dimer_optim_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}_M_${M}
SSResSinglet = ${local_ham_dir}/out/SS/singlet_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}
SSResOriginal = ${local_ham_dir}/out/SS/original_L_[${L},${L}]_J_[${J},1]_T_${T}_N_${N}


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
	python make_ss_local.py -l original -J ${J} -M ${M} -P ${P} > ${LOGFILE}

${SSResDimerOptim}: ${SSOutputDimerOptim}
	cd  ${execute_dir};\
	mkdir -p $(shell dirname ${SSResDimerOptim}); \
	./mpi_bash.sh -f ${SSResDimerOptim} ../Release/main_MPI -m SS2 -ham ${SSOutputDimerOptim}/H -obs ${SSOutputDimerOptim}/Sz -N ${N} -T ${T} \
	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2))

${SSResSinglet}: ${SSOutputSinglet}
	cd  ${execute_dir};\
	mkdir -p $(shell dirname ${SSResSinglet}); \
	./mpi_bash.sh -f ${SSResSinglet} ../Release/main_MPI -m SS2 -ham ${SSOutputSinglet}/H -obs ${SSOutputSinglet}/Sz -N ${N} -T ${T} \
	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2))

${SSResOriginal}: ${SSOutputOriginal}
	cd  ${execute_dir};\
	mkdir -p $(shell dirname ${SSResOriginal}); \
	./mpi_bash.sh -f ${SSResOriginal} ../Release/main_MPI -m SS1 -ham ${SSOutputOriginal}/H -obs ${SSOutputOriginal}/Sz -N ${N} -P1 ${J}  -T ${T} \
	-L1 $$(( $(L)/2)) -L2 $$(( $(L)/2))


.PHONY: clean_worm clean_ham clean

clean_worm:
	rm -f ${SSResDimerOptim} ${SSResSinglet} ${SSResDimerOptim}

clean_ham:
	rm -rf ${SSOutputDimerOptim} ${SSOutputSinglet} ${SSOutputOriginal}


clean: clean_worm clean_ham

PHONY:. SSDimerOptim
PHONY:. SSSinglet
PHONY:. SSOriginal

SSDimerOptim: ${SSResDimerOptim}
SSSinglet: ${SSResSinglet}
SSOriginal: ${SSResOriginal}

PHONY:. SSAll
SSAll:. ${SSResDimerOptim} ${SSResSinglet}

# ${SSResDimerOptim} ${SSResSinglet} ${SSResOriginal} : SSDimerOptim SSSinglet SSOriginal
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