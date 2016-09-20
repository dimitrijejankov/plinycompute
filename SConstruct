# SConstuct
# for PDB
# created on: 12/04/2015

import os
import re
import platform
common_env = Environment(CXX = 'clang++')

SRC_ROOT = os.path.join(Dir('.').abspath, "src") # assume the root source dir is in the working dir named as "src"

# OSX settings
if common_env['PLATFORM'] == 'darwin':
	print 'Compiling on OSX'
  	common_env.Append(CXXFLAGS = '-std=c++1y -Wall -g -O0')
	common_env.Replace(CXX = "clang++")

# Linux settings
elif  common_env['PLATFORM'] == 'posix':
	print 'Compiling on Linux'
	common_env.Append(LIBS = ['libdl.so', 'uuid'])
  	common_env.Append(CXXFLAGS = '-std=c++14 -Wall -g -O0 -ldl')
	common_env.Append(LINKFLAGS = '-pthread')
	common_env.Replace(CXX = "clang++")


# two code files that will be included by the VTableMap to pre-load all of the
# built-in object types into the map
objectTargetDir = os.path.join(SRC_ROOT, 'objectModel', 'headers')
def extractCode (common_env, targets, sources):
	#
	print objectTargetDir + 'BuiltinPDBObjects.h'
	#
	# this is the file where the produced list of includes goes
	includeFile = open(os.path.join(objectTargetDir, 'BuiltinPDBObjects.h'), 'w+')
	includeFile.write ("// Auto-generated by code in SConstruct\n")
	#
	# this is the file where the produced code goes
	codeFile = open(os.path.join(objectTargetDir,'BuiltinPDBObjects.cc'), 'w+')
	codeFile.write ("// Auto-generated by code in SConstruct\n\n")
	codeFile.write ("// first, record all of the type codes\n")
	#
	# this is the file where all of the built-in type codes goes
	typeCodesFile = open(os.path.join(objectTargetDir, 'BuiltInObjectTypeIDs.h'), 'w+')
	typeCodesFile.write ("// Auto-generated by code in SConstruct\n") 
 	typeCodesFile.write ('#define NoMsg_TYPEID 0\n')
	#
	# loop through all of the files in the directory
	counter = 1;
	allClassNames = []
	for fileName in sources:
        	datafile = file(fileName)
		#
		# seaeach foe a line like:
		# // PRELOAD %ObjectTwo%
		p = re.compile('//\s*PRELOAD\s*%[\w\s\<\>]*%')
		for line in datafile:
			#
			# if we found the line
			if p.search(line):
				#
				# add this .h file to the list of files to include
				includeFile.write ('#include "' + fileName + '"\n')
				#
				# extract the text between the two '%' symbols
				m = p.search(line)
				instance = m.group ()
				p = re.compile('%[\w\s\<\>]*%')
				m = p.search(instance)
				classname = (m.group ())[1:-1]
				#
				codeFile.write ('objectTypeNamesList [getTypeName <' + classname + '> ()] = ' + str(counter) + ';\n')
				allClassNames.append (classname)
				#
				# and here we write out the built-in type codes
				pattern = re.compile('\<[\w\s\<\>]*\>')
				if pattern.search (classname):
					templateArg = pattern.search (classname)
					classname = classname.replace (templateArg.group (), "").strip ()
					#
				typeCodesFile.write('#define ' + classname + '_TYPEID ' + str(counter) + '\n')
				counter = counter + 1
				#
	counter = 1
	codeFile.write ('\n// now, record all of the vTables\n')
	for classname in allClassNames:
		#
		codeFile.write ('{\n\tconst UseTemporaryAllocationBlock tempBlock{1024 * 24};');
		codeFile.write ('\n\ttry {\n\t\t')
		codeFile.write (classname + ' tempObject;\n')
		codeFile.write ('\t\tallVTables [' + str(counter) + '] = tempObject.getVTablePtr ();\n')
		codeFile.write ('\t} catch (NotEnoughSpace &e) {\n\t\t')
		codeFile.write ('std :: cout << "Not enough memory to allocate ' + classname + ' to extract the vTable.\\n";\n\t}\n}\n\n');
		counter = counter + 1
	#

# here we get a list of all of the .h files in the 'headers' directory
from os import listdir
from os.path import isfile, isdir, join, abspath
objectheaders = os.path.join(SRC_ROOT, 'builtInPDBObjects', 'headers')
onlyfiles = [abspath(join(objectheaders, f)) for f in listdir(objectheaders) if isfile(join(objectheaders, f)) and f[-2:] == '.h']

# tell scons that the two files 'BuiltinPDBObjects.h' and 'BuiltinPDBObjects.cc' depend on everything in
# the 'headers' directory
common_env.Depends (objectTargetDir + 'BuiltinPDBObjects.h', onlyfiles)
common_env.Depends (objectTargetDir + 'BuiltinPDBObjects.cc', onlyfiles)
common_env.Depends (objectTargetDir + 'BuiltInObjectTypeIDs.h', onlyfiles)

# tell scons that the way to build 'BuiltinPDBObjects.h' and 'BuiltinPDBObjects.cc' is to run extractCode
builtInObjectBuilder = Builder (action = extractCode)
common_env.Append (BUILDERS = {'ExtactCode' : extractCode})
common_env.ExtactCode ([objectTargetDir + 'BuiltinPDBObjects.h', objectTargetDir + 'BuiltinPDBObjects.cc', objectTargetDir + 'BuiltInObjectTypeIDs.h'], onlyfiles)

# Construct a dictionary where each key is the directory basename of a PDB system component folder and each value
# is a list of .cc files used to implement that component.
#
# Expects the path structure of .cc files to be: SRC_ROOT / [component name] / source / [ComponentFile].cc
#
# For example, the structure:
#
#	src/                 <--- assume SRC_ROOT is here
#		compA/
#			headers/
#			source/
#				file1.cc
#				file2.cc
#				readme.txt
#		compB/
#			headers/
#			source/
#				file3.cc
#				file4.cc
#
#
# would result in component_dir_basename_to_cc_file_paths being populated as:
#
#	{'compA':[SRC_ROOT + "/compA/source/file1.cc", SRC_ROOT + "/compA/source/file2.cc"],
#    'compB':[SRC_ROOT + "/compB/source/file3.cc", SRC_ROOT + "/compB/source/file3.cc"]}
#
# on a Linux system.
component_dir_basename_to_cc_file_paths = dict ()
src_root_subdir_paths = [path for path in  map(lambda s: join(SRC_ROOT, s), listdir(SRC_ROOT)) if isdir(path)]
for src_subdir_path in src_root_subdir_paths:

	source_folder = join(src_subdir_path, 'source/')
	if(not isdir(source_folder)): # if no source folder lives under the subdir_path, skip this folder
		continue

	src_subdir_basename = os.path.basename(src_subdir_path)

	# first, map build output folders (on the left) to source folders (on the right)
	common_env.VariantDir(join('build/', src_subdir_basename), [source_folder], duplicate = 0)

	# next, add all of the sources in
	allSources = [abspath(join(join ('build/', src_subdir_basename),f2)) for f2 in listdir(source_folder) if isfile(join(source_folder, f2)) and f2[-3:] == '.cc']
	component_dir_basename_to_cc_file_paths [src_subdir_basename] = allSources


# List of folders with headers
headerpaths = [abspath(join(join(SRC_ROOT, f), 'headers/')) for f in listdir(SRC_ROOT) if os.path.isdir (join(join(SRC_ROOT, f), 'headers/'))]




#boost has its own folder structure, which is difficult to be converted to our headers/source structure --Jia
# set BOOST_ROOT and BOOST_SRC_ROOT
BOOST_ROOT = os.path.join(Dir('.').abspath, "src/boost")
BOOST_SRC_ROOT = os.path.join(Dir('.').abspath, "src/boost/libs")
# map all boost source files to a list
boost_component_dir_basename_to_cc_file_paths = dict ()
boost_src_root_subdir_paths = [path for path in  map(lambda s: join(BOOST_SRC_ROOT, s), listdir(BOOST_SRC_ROOT)) if isdir(path)]
for boost_src_subdir_path in boost_src_root_subdir_paths:
        boost_source_folder = join(boost_src_subdir_path, 'src/')
        if(not isdir(boost_source_folder)): # if no source folder lives under the subdir_path, skip this folder
                continue

        boost_src_subdir_basename = os.path.basename(boost_src_subdir_path)

        # first, map build output folders (on the left) to source folders (on the right)
        common_env.VariantDir(join('build/', boost_src_subdir_basename), [boost_source_folder], duplicate = 0)

        # next, add all of the sources in
        allBoostSources = [abspath(join(join ('build/', boost_src_subdir_basename),f2)) for f2 in listdir(boost_source_folder) if isfile(join(boost_source_folder, f2)) and f2[-4:] == '.cpp']
        boost_component_dir_basename_to_cc_file_paths [boost_src_subdir_basename] = allBoostSources



# append boost to headerpaths
headerpaths.append(BOOST_ROOT)



# Adds header folders and required libraries
common_env.Append(CPPPATH = headerpaths)

print 'Platform: ' + platform.platform()
print 'System: ' + platform.system()
print 'Release: ' + platform.release()
print 'Version: ' + platform.version()

all = [component_dir_basename_to_cc_file_paths['serverFunctionalities'], 
       component_dir_basename_to_cc_file_paths['bufferMgr'],
       component_dir_basename_to_cc_file_paths['communication'],  
       component_dir_basename_to_cc_file_paths['catalog'], 
       component_dir_basename_to_cc_file_paths['pdbServer'], 
       component_dir_basename_to_cc_file_paths['objectModel'], 
       component_dir_basename_to_cc_file_paths['work'], 
       component_dir_basename_to_cc_file_paths['memory'], 
       component_dir_basename_to_cc_file_paths['storage'],
       component_dir_basename_to_cc_file_paths['distributionManager'], 
       boost_component_dir_basename_to_cc_file_paths['filesystem'],
       boost_component_dir_basename_to_cc_file_paths['program_options'],
       boost_component_dir_basename_to_cc_file_paths['smart_ptr'],
       boost_component_dir_basename_to_cc_file_paths['system']]

common_env.SharedLibrary('libraries/libSharedEmployee.so', ['src/sharedLibraries/source/SharedEmployee.cc'] + all)
common_env.SharedLibrary('libraries/libChrisSelection.so', ['src/sharedLibraries/source/ChrisSelection.cc'] + all)
common_env.SharedLibrary('libraries/libStringSelection.so', ['src/sharedLibraries/source/StringSelection.cc'] + all)
common_env.Program('bin/test14', ['src/tests/source/Test14.cc'] + all)
common_env.Program('bin/test15', ['src/tests/source/Test15.cc'] + all)
common_env.Program('bin/test16', ['src/tests/source/Test16.cc'] + all)
common_env.Program('bin/test17', ['src/tests/source/Test17.cc'] + all)
common_env.Program('bin/test18', ['src/tests/source/Test18.cc'] + all)
common_env.Program('bin/test19', ['src/tests/source/Test19.cc'] + all)
common_env.Program('bin/test20', ['src/tests/source/Test20.cc'] + all)
common_env.Program('bin/test21', ['src/tests/source/Test21.cc'] + all)
common_env.Program('bin/test22', ['src/tests/source/Test22.cc'] + all)
common_env.Program('bin/test23', ['src/tests/source/Test23.cc'] + all)
common_env.Program('bin/test24', ['src/tests/source/Test24.cc'] + all)
common_env.Program('bin/test24-temp', ['src/tests/source/Test24-temp.cc'] + all)
common_env.Program('bin/test25', ['src/tests/source/Test25.cc'] + all)
common_env.Program('bin/test26', ['src/tests/source/Test26.cc'] + all)
common_env.Program('bin/test27', ['src/tests/source/Test27.cc'] + all)
common_env.Program('bin/test28', ['src/tests/source/Test28.cc'] + all)
common_env.Program('bin/test29', ['src/tests/source/Test29.cc'] + all)
common_env.Program('bin/test30', ['src/tests/source/Test30.cc'] + all)
common_env.Program('bin/test31', ['src/tests/source/Test31.cc'] + all)
common_env.Program('bin/test1', ['src/tests/source/Test1.cc'] + all)
common_env.Program('bin/test2', ['src/tests/source/Test2.cc'] + all)
common_env.Program('bin/test3', ['src/tests/source/Test3.cc'] + all)
common_env.Program('bin/test4', ['src/tests/source/Test4.cc'] + all)
common_env.Program('bin/test5', ['src/tests/source/Test5.cc'] + all)
common_env.Program('bin/test6', ['src/tests/source/Test6.cc'] + all)
common_env.Program('bin/test7', ['src/tests/source/Test7.cc'] + all)
common_env.Program('bin/test8', ['src/tests/source/Test8.cc'] + all)
common_env.Program('bin/test9', ['src/tests/source/Test9.cc'] + all)
common_env.Program('bin/test10', ['src/tests/source/Test10.cc'] + all)
common_env.Program('bin/test11', ['src/tests/source/Test11.cc'] + all)
common_env.Program('bin/test12', ['src/tests/source/Test12.cc'] + all)
common_env.Program('bin/test13', ['src/tests/source/Test13.cc'] + all)
common_env.Program('bin/test100', ['src/tests/source/Test100.cc'] + all)
common_env.Program('bin/test600', ['src/tests/source/Test600.cc'] + all)
common_env.Program('bin/pdbServer', ['src/mainServer/source/PDBMainServerInstance.cc'] + all)
common_env.Program('bin/getListNodesTest', ['src/tests/source/GetListNodesTest.cc'] + all)