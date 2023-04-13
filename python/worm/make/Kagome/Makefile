# Define variables
execute_dir := /home/user/project/build
python_dir := /home/user/project/python
local_ham_dir := ${python_dir}/rmsKit
file_dir := $(shell pwd)
LOGPATH := $(file_dir)/logs
LOGFILE := $(LOGPATH)/$(shell date --iso=seconds).txt

# Define default values for optional variables
N ?= 100000
T ?= 1.0
M ?= 120
P ?= 12
L1 ?= 2
L2 ?= 2
Jx ?= 1
Jy ?= 1
Jz ?= 1
Hx ?= 0
Hz ?= 0

# Define array and result paths
array_name := Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_hx_${Hx}_hz_${Hz}
res_folder := ${L1}x${L2}


Array3siteDiag := ${local_ham_dir}/array/KH/3siteDiag/none/${array_name}
Res3siteDiag := ${file_dir}/out/KH/3siteDiag/${res_folder}/${array_name}/T_${T}_N_${N}.txt
Array3site := ${local_ham_dir}/array/KH/3site/none/${array_name}
Res3site := ${file_dir}/out/KH/3site/none/${res_folder}/${array_name}/T_${T}_N_${N}.txt

Array3siteMes := ${local_ham_dir}/array/KH/3site/mes/${array_name}/M_${M}
Res3siteMes := ${file_dir}/out/KH/3site/mes/${res_folder}/${array_name}/T_${T}_N_${N}.txt
# Define phony targets

# # * define targets
${Array3site}: 
	cd ${local_ham_dir};\
	python optimize.py -m KH -u 3site -Jz ${Jz} -Jx ${Jx} -Jy ${Jy} -Hx ${Hx} -Hz ${Hz} >> ${LOGFILE};\

${Res3site}: ${Array3site}
	cd ${execute_dir}; \
	mkdir -p $(shell dirname ${Res3site}); \
	./main_MPI -m Kagome3 -ham ${Array3site}/H -N ${N} -T ${T} -L1 ${L2} -L2 ${L2} > ${Res3site}

${Array3siteMes}: 
	cd ${local_ham_dir};\
	python optimize.py -m KH -u 3site -Jz ${Jz} -Jx ${Jx} -Jy ${Jy} -Hx ${Hx} -Hz ${Hz} -M ${M} -P ${P} -l mes  >> ${LOGFILE}


${Res3siteMes}: ${Array3siteMes}
	cd ${execute_dir}; \
	mkdir -p $(shell dirname ${Res3siteMes}); \
	./main_MPI -m Kagome3 -ham ${Array3siteMes}/H -N ${N} -T ${T} -L1 ${L2} -L2 ${L2} > ${Res3siteMes}


${Array3siteDiag}:
	cd ${local_ham_dir};\
	python optimize.py -m KH -u 3siteDiag -Jz ${Jz} -Jx ${Jx} -Jy ${Jy} -Hx ${Hx} -Hz ${Hz} >> ${LOGFILE}

${Res3siteDiag}: ${Array3siteDiag}
	cd ${execute_dir}; \
	mkdir -p $(shell dirname ${Res3siteDiag}); \
	./main_MPI -m Kagome3 -ham ${Array3siteDiag}/H -N ${N} -T ${T} -L1 ${L2} -L2 ${L2} > ${Res3siteDiag}

.PHONY: all 3Site 3SiteMes 3SiteDiag Array3siteMes Array3siteDiag Array3site
3Site: ${Res3site}
3SiteMes: ${Res3siteMes}
3SiteDiag: ${Res3siteDiag}

Array3siteMes : ${Array3siteMes}
Array3siteDiag : ${Array3siteDiag}
Array3site : ${Array3site}

# ARRAY = Array3siteMes Array3siteDiag Array3site
# Array3siteMes = ${Array3siteMes}
# $(ARRAY_SITES) : % : $$(%)
# 	echo "Array site $@"

# all: 3Site 3SiteMes 3SiteDiag
