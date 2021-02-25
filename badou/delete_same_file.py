# coding=utf-8
import os
from pbxproj import XcodeProject

# default，主路径填写工程主路径，或者脚本放在工程主路径
# project_main_path = sys.path[0]
project_main_path = '/Users/cody/Desktop/iOS/talker/Talker-iOS/Talker/Talker'
project_pxb_path = '/Users/cody/Desktop/iOS/talker/Talker-iOS/Talker/Talker.xcodeproj/project.pbxproj'
# 主端文件集合
local_str_path = project_main_path
# ULPikaHomeNavigationView
# 组件文件集合
shareFeatures_file_set = set()
# 主端文件集合，key->文件名，value->文件路径
main_file_dict = {}
# 要搜索的路径，递归搜
str_dir_path = project_main_path + '/ULShareFeatures/ULVideo'

def findFromFile(path):
	paths = os.listdir(path)
	for aCompent in paths:
		aPath = os.path.join(path, aCompent)
		if os.path.isdir(aPath):
			findFromFile(aPath)
		elif os.path.isfile(aPath) and (os.path.splitext(aPath)[1]=='.m' or os.path.splitext(aPath)[1]=='.h'):
			shareFeatures_file_set.add(os.path.basename(aPath))

def getLocalFileDict(path):
	paths = os.listdir(path)
	for aCompent in paths:
		aPath = os.path.join(path, aCompent)
		if 'ULShareFeatures' in aPath:
			continue
		if os.path.isdir(aPath):
			getLocalFileDict(aPath)
		elif os.path.isfile(aPath) and (os.path.splitext(aPath)[1] == '.m' or os.path.splitext(aPath)[1] == '.h'):
			if main_file_dict.get(aPath,-1) == -1:
				main_file_dict[os.path.basename(aPath)] = aPath

def deleteDuplicateFileInMain():
	# project = XcodeProject.load(project_pxb_path)
	for fileName in shareFeatures_file_set:
		if fileName in main_file_dict.keys():
			# res = project.remove_files_by_path(main_file_dict[fileName])
			os.remove(main_file_dict[fileName])
	# project.save()

if __name__ == '__main__':
	findFromFile(str_dir_path)
	getLocalFileDict(local_str_path)
	deleteDuplicateFileInMain()
