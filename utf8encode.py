import glob

def utf8encode(file_path):
	'''
	encodes a file to utf-8
	'''	
	with open(file_path, 'r+', encoding='utf-8', errors = 'replace') as file:				
		file.write(file.read())
		file.truncate()
	
	print(f'Converted {file_path} utf-8 (unicode)')
	return	

def get_html_paths(roots):	
	'''
	gets path of html files recursively starting from root in roots(list of roots)
	'''
	assert isinstance(roots,(tuple,list,set))
	all_files = []
	for root in roots:
	    files = glob.glob(f'{root}/**/*.html',
	                      recursive=True)
	    all_files += files if type(files) != str else [files]

	return all_files

if __name__ == '__main__':	
	'''
	converts .html files in /docs to utf-8 and unicode charset
	'''
	paths = get_html_paths(['docs'])
	for path in paths:		
		utf8encode(path)
