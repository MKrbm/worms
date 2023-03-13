# Define variables
execute_dir := /home/user/project/build
python_dir := /home/user/project/python
local_ham_dir := ${python_dir}/make_local
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
.PHONY: all 3Site 3SiteMes 3SiteDiag Array3siteMes

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

3Site: ${Res3site}
3SiteMes: ${Res3siteMes}
3SiteDiag: ${Res3siteDiag}
Array3siteMes : ${Array3siteMes}
all: 3Site 3SiteMes 3SiteDiag

# Define all target

# # Define rule to create array files
# ${Array3site} ${Array3siteMes} ${Array3siteDiag}:
# 	cd ${local_ham_dir}; python optimize.py -m KH -u 3site${@F:5} -Jz ${Jz} -Jx ${Jx} -Jy ${Jy} -Hx ${Hx} -Hz ${Hz} ${@F:0:3} > ${LOGFILE}

# # Define rule to create result files
# ${Res3site} ${Res3siteMes} ${Res3siteDiag}: ${Array3site}${@F:9}
# 	mkdir -p $(shell dirname $@); cd ${execute_dir}; ./main_MPI -m Kagome3 -ham $</H -N ${N} -T ${T} -L1 ${L2} -L2 ${L2} > $@

# # Define 3Site target
# 3Site: ${Res3site}

# # Define 3SiteMes target
# 3SiteMes: ${Res3siteMes}

# # Define 3SiteDiag target
# 3SiteDiag: ${Res3siteDiag}



# # make_local := ./make_local
# execute_dir := /home/user/project/build
# python_dir := /home/user/project/python
# # call_dir := /home/user/project/python/worm
# local_ham_dir := ${python_dir}/make_local
# file_dir := $(shell pwd)

# #* for log files
# LOGPATH = $(file_dir)/logs
# LOGFILE = $(LOGPATH)/$(shell date --iso=seconds)

# #* for Kagome
# #* set default values for optional variables if not defined
# N ?= 100000
# T ?= 1.0
# M ?= 120
# P ?= 12

# ifndef L1 
# $(warning L1 is not defined, using default value of 2)
# L1 := 2
# endif

# ifndef L2
# $(warning L2 is not defined, using default value of 2)
# L2 := 2
# endif

# ifndef Jx
# $(warning Jx is not defined, using default value of 1)
# Jx := 1
# endif

# ifndef Jy
# $(warning Jy is not defined, using default value of 1)
# Jy := 1
# endif

# ifndef Jz
# $(warning Jz is not defined, using default value of 1)
# Jz := 1
# endif

# ifndef Hx
# $(warning Jx is not defined, using default value of 1)
# Hx := 0
# endif

# ifndef Hz
# $(warning Jy is not defined, using default value of 1)
# Hz := 0
# endif


# array_name = Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_hx_${Hx}_hz_${Hz}
# res_folder = ${L1}x${L2}

# Array3siteMes = ${local_ham_dir}/array/KH/3site/mes/${array_name}
# Res3siteMes = ${file_dir}/out/KH/3site/mes/${res_folder}/${array_name}/T_${T}_N_${N}.txt

# Array3siteDiag = ${local_ham_dir}/array/KH/3siteDiag/none/${array_name}
# Res3siteDiag = ${file_dir}/out/KH/3siteDiag/${res_folder}/${array_name}/T_${T}_N_${N}.txt

# Array3site = ${local_ham_dir}/array/KH/3site/none/${array_name}
# Res3site = ${file_dir}/out/KH/3site/none/${res_folder}/${array_name}/T_${T}_N_${N}.txt


# # * define targets
# ${Array3site}: 
# 	cd ${local_ham_dir};\
# 	python optimize.py -m KH -u 3site -Jz ${Jz} -Jx ${Jx} -Jy ${Jy} -Hx ${Hx} -Hz ${Hz} > ${LOGFILE};\


# ${Array3siteMes}: 
# 	cd ${local_ham_dir};\
# 	python optimize.py -m KH -u 3site -Jz ${Jz} -Jx ${Jx} -Jy ${Jy} -Hx ${Hx} -Hz ${Hz} -M ${M} -P ${P} -l mes  > ${LOGFILE};\

# ${Array3siteDiag}:
# 	cd ${local_ham_dir};\
# 	python optimize.py -m KH -u 3siteDiag -Jz ${Jz} -Jx ${Jx} -Jy ${Jy} -Hx ${Hx} -Hz ${Hz} > ${LOGFILE};\


# ${Res3site}: ${Array3site}
# 	cd ${execute_dir}; \
# 	mkdir -p $(shell dirname ${Res3site}); \
# 	./main_MPI -m Kagome3 -ham ${Array3site}/H -N ${N} -T ${T} -L1 ${L2} -L2 ${L2} > ${Res3site}

# ${Res3siteMes}: ${Array3siteMes}
# 	cd ${execute_dir}; \
# 	mkdir -p $(shell dirname ${Res3siteMes}); \
# 	./main_MPI -m Kagome3 -ham ${Array3siteMes}/H -N ${N} -T ${T} -L1 ${L2} -L2 ${L2} > ${Res3siteMes}

# ${Res3siteDiag}: ${Array3siteDiag}
# 	cd ${execute_dir}; \
# 	mkdir -p $(shell dirname ${Res3siteDiag}); \
# 	./main_MPI -m Kagome3 -ham ${Array3siteDiag}/H -N ${N} -T ${T} -L1 ${L2} -L2 ${L2} > ${Res3siteDiag}

# PHONY:. 3Site
# PHONY:. 3SiteMes
# PHONY:. 3SiteDiag

# 3Site: ${Res3site}
# 3SiteMes: ${Res3siteMes}
# 3SiteDiag: ${Res3siteDiag}
