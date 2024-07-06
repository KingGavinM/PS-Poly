# External Imports
import os as _os
import numpy as _np
import matplotlib.pyplot as _plt
from datetime import date as _datetime
from shutil import rmtree as _rmtree

# PS-Poly Imports
from . import filters
from . import length
from . import utilities

'''
Contents
----------------------------------------------------------------------------------------------------------------------------------------------=
1. Basic Methods
2. Advanced Methods
3. User-interface Methods
'''

'''
1. Basic Methods
-----------------------------------------------------------------------------------------------------------------------------------------------
Data handling and polydat object definition
'''

pspoly_home = _os.getcwd()
pspoly_save_option = 'load'
def initialize(home=pspoly_home,save_option=pspoly_save_option):
	global pspoly_home
	pspoly_home = home
	global pspoly_save_option
	pspoly_save_option = save_option

def load(address):
	try:
		info = open(_os.path.join(pspoly_home,address,'info.txt')); lines = info.readlines(); info.close()
		pdat = polydat(lines[3][:-1],_os.path.join(pspoly_home,address,'data.npy'),float(lines[7]),root=lines[5][:-1],save=False)
		pdat._date = lines[1][:-1]
		return pdat
	except:
		print(f'WARNING: A polydat object \'{address}\' could not be loaded.')

class polydat:
	def __init__(self,name,img,px_size,scale_factor=1,root='',save=True):

		self._name = name
		self._root = root

		self._px_size = px_size / scale_factor
		self._date = str(_datetime.today())

		if type(img) == str:
			if _os.path.splitext(img)[-1] == '.npy': temp_img = _np.load(img)
			else: temp_img = utilities.to_gray(_plt.imread(img))
		else: temp_img = img

		if scale_factor != 1:
			self._img = _np.zeros((scale_factor*temp_img.shape[0],scale_factor*temp_img.shape[1]))
			for iy in range(self._img.shape[0]):
				for ix in range(self._img.shape[1]):
					self._img[iy,ix] = temp_img[iy//scale_factor,ix//scale_factor]
		else: self._img = temp_img

		if save: self.save_dat()


	# Saves the polydat object to disk. Runs if the save option is set to True.
	def save_dat(self):
		if not _os.path.exists(self.get_path()):
			_os.mkdir(self.get_path())
			info = open(self.get_infpath(),'w'); info.write(f'Creation Date\n{self.get_date()}\nName\n{self.get_name()}\nRoot\n{self.get_root()}\nPixel Size\n{self.get_px_size()}'); info.close()
			_np.save(self.get_datpath(),self.get_img())
		elif pspoly_save_option == 'ask':
			option = input(f'WARNING: A polydat object \'{self.get_id()}\' already exists.\nWould you like to overwrite it? (y/n): ')
			if option == 'y':
				print(f'An old polydat object \'{self.get_id()}\' has been overwritten.')
				_rmtree(self.get_path())
				_os.mkdir(self.get_path())
				info = open(self.get_infpath(),'w'); info.write(f'Creation Date\n{self.get_date()}\nName\n{self.get_name()}\nRoot\n{self.get_root()}\nPixel Size\n{self.get_px_size()}'); info.close()
				_np.save(self.get_datpath(),self.get_img())
			elif option == 'n':
				print(f'An old polydat object \'{self.get_id()}\' has been loaded.')
				self = load(self.get_id())
			else:
				print(f'WARNING: An invalid option has been selected. Loading an old polydat object \'{self.get_id()}\'.')
				self = load(self.get_id())
		elif pspoly_save_option == 'overwrite':
			print(f'WARNING: An old polydat object \'{self.get_id()}\' has been overwritten.')
			_rmtree(self.get_path())
			_os.mkdir(self.get_path())
			info = open(self.get_infpath(),'w'); info.write(f'Creation Date\n{self.get_date()}\nName\n{self.get_name()}\nRoot\n{self.get_root()}\nPixel Size\n{self.get_px_size()}'); info.close()
			_np.save(self.get_datpath(),self.get_img())
		elif pspoly_save_option == 'load':
			print(f'WARNING: An old polydat object \'{self.get_id()}\' has been loaded.')
			self = load(self.get_id())
		else:
			print(f'WARNING: An invalid save option has been selected. Loading an old polydat object \'{self.get_id()}\'.')
			self = load(self.get_id())

	# Saves the bnd argument to a file given by get_bndpath(). The bnd argument should take the form ((miny,maxy),(minx,maxx)).
	def save_bnd(self,bnd):
		bounds = open(self.get_bndpath(),'w'); bounds.write(f'{bnd[0][0]}\n{bnd[0][1]}\n{bnd[1][0]}\n{bnd[1][1]}'); bounds.close()

	# Runs automatically after preparing the polydat. Indicates that the polydat is ready for use with the analyze method.
	def verify(self):
		vfile = open(self.get_vpath(),'w'); vfile.write('This polydat object is ready for analysis.'); vfile.close()

	def get_bnd(self):
		try:
			bounds = open(self.get_bndpath()); lines = bounds.readlines(); bounds.close()
			return ((int(lines[0]), int(lines[1])), (int(lines[2]), int(lines[3])))
		except: return None

	def get_name(self):
		return self._name
	
	def get_img(self):
		return _np.copy(self._img)

	def get_px_size(self):
		return self._px_size

	def get_date(self):
		return self._date

	# Returns the path to the polydat.
	def get_path(self):
		if self._root == '':
			return  _os.path.join(pspoly_home,self._name)
		else:
			return _os.path.join(pspoly_home,self._root,self._name)

	# Returns the path containing the polydat.
	def get_superpath(self):
		if self._root == '':
			return pspoly_home
		else:
			return _os.path.join(pspoly_home,self._root)

	# Returns the path to the data.npy file of the polydat.
	def get_datpath(self):
		if self._root == '':
			return _os.path.join(pspoly_home,self._name,'data.npy')
		else:
			return _os.path.join(pspoly_home,self._root,self._name,'data.npy')

	# Returns the path to the info.txt file of the polydat.
	def get_infpath(self):
		if self._root == '':
			return _os.path.join(pspoly_home,self._name,'info.txt')
		else:
			return _os.path.join(pspoly_home,self._root,self._name,'info.txt')

	# Returns the path to the bounds.txt file of the polydat.
	def get_bndpath(self):
		if self._root == '':
			return _os.path.join(pspoly_home,self._name,'bounds.txt')
		else:
			return _os.path.join(pspoly_home,self._root,self._name,'bounds.txt')

	# Returns the path to the verify.txt file of the polydat.
	def get_vpath(self):
		if self._root == '':
			return _os.path.join(pspoly_home,self._name,'verify.txt')
		else:
			return _os.path.join(pspoly_home,self._root,self._name,'verify.txt')

	# Returns the name of the top-level polydat object in which the polydat is nested.
	def get_oname(self):
		path = self.get_path()

		while True:
			ht = _os.path.split(path)
			if ht[0] == pspoly_home:
				return ht[1]
			else:
				path = ht[0]

	def get_root(self):
		return self._root

	# Returns the path to the polydat, excluding pspoly_home. Useful for the load method.
	def get_id(self):
		if self._root == '':
			return self._name
		else:
			return _os.path.join(self._root,self._name)

	# Creates a polydat object nested within the polydat.
	def subdat(self,name,img,px_size,scale_factor=1,save=True):
		if self._root == '':
			root = self._name
		else:
			root = _os.path.join(self._root,self._name)
		return polydat(name,img,px_size,scale_factor,root,save)
		
	# Lists all folders within the polydat. The folder names are equivalent to the names of the nested polydat objects, unless a subfolder has been created manually.
	def get_subdata(self,style='dict'):
		path = self.get_path()
		ldir = _os.listdir(path)
		subdata = [load(_os.path.join(self.get_id(),f)) for f in ldir if _os.path.isdir(_os.path.join(path,f))]
		if style == 'list': return subdata
		else: return {dat.get_name():dat for dat in subdata}

	def isolate_skeleton(self,particle):
		return load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))

	def get_particle(self,particle,imtype='',pad=0):
		if imtype == 'mask':
			mask = load(_os.path.join(self.get_id(),'mask'))
			shp = self.get_img().shape
			pdat = load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))
			bnds = pdat.get_bnd()
			return mask.get_img()[max(0,bnds[0][0]-pad):min(bnds[0][1]+1+pad,shp[0]),max(0,bnds[1][0]-pad):min(bnds[1][1]+1+pad,shp[1])]
		elif imtype == 'skeleton':
			skel = load(_os.path.join(self.get_id(),'mask','skeleton'))
			shp = self.get_img().shape
			pdat = load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))
			bnds = pdat.get_bnd()
			return skel.get_img()[max(0,bnds[0][0]-pad):min(bnds[0][1]+1+pad,shp[0]),max(0,bnds[1][0]-pad):min(bnds[1][1]+1+pad,shp[1])]
		elif imtype == 'points':
			skel = load(_os.path.join(self.get_id(),'mask','skeleton'))
			shp = self.get_img().shape
			pdat = load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))
			bnds = pdat.get_bnd()

			end_points = load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}','end_points')).get_img()
			branch_points = load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}','branch_points')).get_img()
			bundle_points = load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}','bundle_points')).get_img()

			return skel.get_img()[max(0,bnds[0][0]-pad):min(bnds[0][1]+1+pad,shp[0]),max(0,bnds[1][0]-pad):min(bnds[1][1]+1+pad,shp[1])] + 3*end_points + branch_points + bundle_points
		else:
			shp = self.get_img().shape
			pdat = load(_os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))
			bnds = pdat.get_bnd()
			return self.get_img()[max(0,bnds[0][0]-pad):min(bnds[0][1]+1+pad,shp[0]),max(0,bnds[1][0]-pad):min(bnds[1][1]+1+pad,shp[1])]

# Returns pspoly_home.
def get_home():
	return pspoly_home

# Returns pspoly_load_option.
def get_save_option():
	return pspoly_save_option

'''
Advanced Methods
-----------------------------------------------------------------------------------------------------------------------------------------------
Features used in the PS-Poly algorithm that the end-user need not implement directly
May not be optimized for ease of use and may not accept high-level polydat inputs
'''

def threshold(dat,threshold_function=filters.threshold,t=None,save=True):
	if t == None: return dat.subdat('mask',threshold_function(dat.get_img()),dat.get_px_size(),save=save)
	else: return dat.subdat('mask',threshold_function(dat.get_img(),t),dat.get_px_size(),save=save)

# Creates a skeletonized subdat of the original polydat object. The input must be a binary mask.
def skeletonize(dat,skeleton_function=filters.skeletonize,save=True):
	return dat.subdat('skeleton',skeleton_function(dat.get_img()),dat.get_px_size(),save=save)

# Creates a subdat for every connected region within the polydat. The input must be a binary mask.
def separate_particles(dat):
	labeled, bounds = filters.label_bound(dat.get_img())
	n = len(labeled)

	particles = []
	for i in range(n):
		particles.append(dat.subdat(f'particle-{i+1}',labeled[i],dat.get_px_size()))
		particles[i].save_bnd(bounds[i])

	return particles
	
# Lists all subdat names beginning with 'particle'. These should be the subdat objects created by the separate_particles method.
def list_particles(dat):
	subdata = dat.get_subdata('list')
	index = 0
	while True:
		subsubdata = []
		for d in subdata[index:]:
			subsubdata.extend(d.get_subdata('list'))
		l = len(subsubdata)
		index += l
		subdata.extend(subsubdata)
		if l == 0: break
	return [d for d in subdata if d.get_name()[:8] == 'particle']

def identify_points(dat):
	points_tuple = filters.mainpoints(dat.get_img())

	end_points = dat.subdat('end_points',points_tuple[0],dat.get_px_size())
	branch_points = dat.subdat('branch_points',points_tuple[1],dat.get_px_size())
	bundle_points = dat.subdat('bundle_points',points_tuple[2],dat.get_px_size())
	branch_points2 = dat.subdat('branch_points2',points_tuple[3],dat.get_px_size())

	return {'end_points':end_points, 'branch_points':branch_points, 'bundle_points':bundle_points, 'branch_points2':branch_points2}

# Returns an array of images corresponding to each subsection of the polydat, separated by branch points. The input must have identified points.
def split_particle(dat):
	img = dat.get_img()
	path = dat.get_id()

	end_points = load(_os.path.join(path,'end_points')).get_img()
	branch_points = load(_os.path.join(path,'branch_points')).get_img()
	bundle_points = load(_os.path.join(path,'bundle_points')).get_img()
	branch_points2 = load(_os.path.join(path,'branch_points2')).get_img()

	branch_points = utilities.relu(branch_points-bundle_points)

	labeled_img = filters._label(utilities.relu(img-bundle_points-branch_points-branch_points2))

	particles = []
	for v in range(_np.max(labeled_img)):
		particles.append((labeled_img == v+1).astype('int'))

	bc = _np.array(_np.where(branch_points == 1)).T
	b2c = _np.array(_np.where(branch_points2 == 1)).T
	bdc = _np.array(_np.where(bundle_points == 1)).T

	# This is allowing multiple areas to be added back at once
	for particle in particles:
		particlep = _np.pad(particle,1)
		particlepc = _np.copy(particlep)
		for i in range(b2c.shape[0]):
			if _np.sum(particlep[b2c[i,0]:b2c[i,0]+3,b2c[i,1]:b2c[i,1]+3]) > 0:
				particle[b2c[i,0],b2c[i,1]] = particlepc[b2c[i,0]+1,b2c[i,1]+1] = 1
		particlep = _np.copy(particlepc)
		for i in range(b2c.shape[0]):
			if particlep[b2c[i,0]+1,b2c[i,1]+2] + particlep[b2c[i,0],b2c[i,1]+1] + particlep[b2c[i,0]+1,b2c[i,1]] + particlep[b2c[i,0]+2,b2c[i,1]+1] > 0:
				particle[b2c[i,0],b2c[i,1]] = particlepc[b2c[i,0]+1,b2c[i,1]+1] = 1
		particlep = _np.copy(particlepc)
		for i in range(bc.shape[0]):
			if _np.sum(particlep[bc[i,0]:bc[i,0]+3,bc[i,1]:bc[i,1]+3]) > 0:
				particle[bc[i,0],bc[i,1]] = 1
		for i in range(bdc.shape[0]):
			if _np.sum(particlep[bdc[i,0]:bdc[i,0]+3,bdc[i,1]:bdc[i,1]+3]) > 0:
				particle[bdc[i,0],bdc[i,1]] = 1

	composite = sum(particles)
	labeled_secondary = filters._label(utilities.relu(branch_points2-composite))

	particles_missed = []
	for v in range(_np.max(labeled_secondary)):
		particles_missed.append((labeled_secondary == v+1).astype('int'))

	for particle in particles_missed:
		particlep = _np.pad(particle,1)
		for i in range(bc.shape[0]):
			if _np.sum(particlep[bc[i,0]:bc[i,0]+3,bc[i,1]:bc[i,1]+3]) > 0:
				particle[bc[i,0],bc[i,1]] = 1
		for i in range(bdc.shape[0]):
			if _np.sum(particlep[bdc[i,0]:bdc[i,0]+3,bdc[i,1]:bdc[i,1]+3]) > 0:
				particle[bdc[i,0],bdc[i,1]] = 1

	particles.extend(particles_missed)

	composite = sum(particles)
	labeled_bundle = filters._label(utilities.relu(bundle_points-composite))

	particles_missed = []
	for v in range(_np.max(labeled_bundle)):
		particles_missed.append((labeled_bundle == v+1).astype('int'))

	particles.extend(particles_missed)

	particles_missed = []
	for x in range(len(bc)):
		for y in range(x+1,len(bc)):
			if abs(bc[y][0]-bc[x][0]) <= 1 and abs(bc[y][1]-bc[x][1]) <= 1:
				missed = _np.zeros(branch_points.shape)
				missed[bc[x][0],bc[x][1]] = missed[bc[y][0],bc[y][1]] = 1
				particles_missed.append(missed)
		for y in range(len(bdc)):
			if abs(bdc[y][0]-bc[x][0]) <= 1 and abs(bdc[y][1]-bc[x][1]) <= 1:
				missed = _np.zeros(branch_points.shape)
				missed[bc[x][0],bc[x][1]] = missed[bdc[y][0],bdc[y][1]] = 1
				particles_missed.append(missed)

	particles.extend(particles_missed)

	return particles

# Completes all necessary steps prior to analyzing the polydat. Returns a skeletonized polydat, which will be the input to the analyze method. Will only return the skeleton if the polydat has been verified.
def prepare(dat):
	if not _os.path.isfile(dat.get_vpath()):
		mask = threshold(dat)
		skeleton = skeletonize(mask)
		particles = separate_particles(skeleton)
		for particle in particles: identify_points(particle)
		dat.verify()
		return skeleton
	else:
		print(f'WARNING: An old polydat object has been loaded instead of preparing \'{dat.get_id()}\' again.')
		if dat.get_root() == '':
			return load(_os.path.join(dat.get_name(),'mask','skeleton'))
		else:
			return load(_os.path.join(dat.get_root(),dat.get_name(),'mask','skeleton'))

# Returns an image indicating the height of every position in the polydat. The input must be a binary mask.
def measure_height(dat):
	bnds = dat.get_bnd()
	himg = load(dat.get_oname()).get_img()
	shp = himg.shape

	if bnds == None: return dat.get_img()*himg
	else: return dat.get_img()*himg[max(0,bnds[0][0]):min(bnds[0][1]+1,shp[0]),max(0,bnds[1][0]):min(bnds[1][1]+1,shp[1])]

# Returns the mean height of all active points in the polydat. The input must be a binary mask.
def mean_height(dat):
	skeleton = dat.get_img()
	sheight = measure_height(dat)
	number = 0
	total = 0

	for iy in range(skeleton.shape[0]):
		for ix in range(skeleton.shape[1]):
			if skeleton[iy,ix] == 1:
				number += 1
				total += sheight[iy,ix]

	return total / number

# Classifies the polydat based on structure and height.
def classify(dat,threshold=None,noise=0.8):
	path = dat.get_id()

	end_points = load(_os.path.join(path,'end_points')).get_img()
	branch_points = load(_os.path.join(path,'branch_points')).get_img()
	bundle_points = load(_os.path.join(path,'bundle_points')).get_img()

	branch_points = utilities.relu(branch_points-bundle_points)

	skeleton = dat.get_img()
	height = measure_height(dat)
	if threshold == None: high = skeleton*0
	else: high = skeleton*(height > threshold)

	nodes = _np.sum(end_points) + _np.sum(branch_points) + _np.max(filters._label(bundle_points,connectivity=1))

	if _np.sum(high) == 0:
		if nodes == 0: return 'looped'
		else:
			particles = split_particle(dat)
			edges = len(particles)
			if nodes-edges == 1:
				if nodes == 2: return 'linear'
				else: return 'branched'
			else: return 'branched and looped'
	elif _np.sum(high) / _np.sum(skeleton) > noise: return 'noise particle'
	elif _np.sum(high*(branch_points + bundle_points)) > 0: return 'overlapped'
	else:
		n = _np.max(filters._label(high))
		if n == 1:
			if nodes == 0: return 'looped with 1 high point'
			else:
				particles = split_particle(dat)
				edges = len(particles)
				if nodes-edges == 1:
					if nodes == 2: return 'linear with 1 high point'
					else: return 'branched with 1 high point'
				else: return 'branched and looped with 1 high point'
		else:
			if nodes == 0: return f'looped with {n} high points'
			else:
				particles = split_particle(dat)
				edges = len(particles)
				if nodes-edges == 1:
					if nodes == 2: return f'linear with {n} high points'
					else: return f'branched with {n} high points'
				else: return f'branched and looped with {n} high points'

def display_results(info):
	stats = info[0]
	n_high = info[1]
	n_noise = info[2]
	Lp = info[3]

	print(f'Particle Type\t\tNumber of Features\t\tAverage length (nm)\t\tPercentage of Total Polymerization Length\nLinear\t\t\t\t{stats[0][0]}\t\t\t\t{stats[0][1]}\t\t\t\t{stats[0][2]}\nLooped\t\t\t\t{stats[1][0]}\t\t\t\t{stats[1][1]}\t\t\t\t{stats[1][2]}\nBranched (no looping)\t\t{stats[2][0]}\t\t\t\t{stats[2][1]}\t\t\t\t{stats[2][2]}\nBranched (with looping)\t\t{stats[3][0]}\t\t\t\t{stats[3][1]}\t\t\t\t{stats[3][2]}\nOverlapped\t\t\t{stats[4][0]}\t\t\t\t{stats[4][1]}\t\t\t\t{stats[4][2]}\n')
	print(f'Number of High Points: {n_high}\nNumber of Noise Particles: {n_noise}\n')
	print(f'Persistence Length (nm): {Lp}')

# Analyzes the polydat. Returns the data as a nested array of strings if the display option is False, otherwise displays the data in the terminal.
def analyze(dat,display=True):
	polys = list_particles(dat)
	threshold = 1.5*mean_height(dat)
	
	n_linear = 0
	l_linear = 0
	n_looped = 0
	l_looped = 0
	n_branched = 0
	l_branched = 0
	n_branched_looped = 0
	l_branched_looped = 0
	n_overlapped = 0
	l_overlapped = 0
	n_high = 0
	n_noise = 0

	linears = []

	for poly in polys:
		shape = classify(poly,threshold)
		if shape[:6] == 'linear':
			n_linear += 1
			l_linear += length.contour(poly)
			linears.append(poly)
		elif shape[:6] == 'looped':
			n_looped += 1
			l_looped += length.contour(poly)
		elif shape[:8] == 'branched':
			if shape[:19] == 'branched and looped':
				n_branched_looped += 1
				l_branched_looped += length.contour(poly)
			else:
				n_branched += 1
				l_branched += length.contour(poly)
		elif shape[:10] == 'overlapped':
			skeleton = poly.get_img()
			height = measure_height(poly)
			high = polydat('',skeleton*(height > threshold),poly.get_px_size(),save=False)

			n_overlapped += 1
			l_overlapped += length.contour(poly) + length.contour(high)
		elif shape == 'noise particle':
			n_noise += 1
		if 'high point' in shape:
			numbers = [int(s) for s in shape.split() if s.isdigit()]
			n_high += numbers[0]

	l_total = l_linear + l_looped + l_branched + l_branched_looped + l_overlapped

	try:
		p_linear = f'{l_linear/n_linear:.1f}'
	except:
		p_linear = 'N/A'
	try:
		p_looped = f'{l_looped/n_looped:.1f}'
	except:
		p_looped = 'N/A'
	try:
		p_branched = f'{l_branched/n_branched:.1f}'
	except:
		p_branched = 'N/A'
	try:
		p_branched_looped = f'{l_branched_looped/n_branched_looped:.1f}'
	except:
		p_branched_looped = 'N/A'
	try:
		p_overlapped = f'{l_overlapped/n_overlapped:.1f}'
	except:
		p_overlapped = 'N/A'

	stats = ((f'{n_linear}',p_linear,f'{100*l_linear/l_total:.1f}%'), (f'{n_looped}',p_looped,f'{100*l_looped/l_total:.1f}%'), (f'{n_branched}',p_branched,f'{100*l_branched/l_total:.1f}%'), (f'{n_branched_looped}',p_branched_looped,f'{100*l_branched_looped/l_total:.1f}%'), (f'{n_overlapped}',p_overlapped,f'{100*l_overlapped/l_total:.1f}%'))
	Lp = f'{length.Lp(linears):.1f}'

	info = (stats, f'{n_high}', f'{n_noise}', Lp)

	if display: display_results(info)
	else: return info

'''
User-interface Methods
-----------------------------------------------------------------------------------------------------------------------------------------------
Implementations of the PS-Poly alogorithm designed for use by the end-user
'''

# Runs the full process on the polydat.
def run(img,px_size,scale_factor=1,name=None,display=True):
	if name == None:
		pdat = polydat('pspoly_temp',img,px_size,scale_factor,'',True)
		skeleton = prepare(pdat)
		info = analyze(skeleton,False)
		_rmtree(pdat.get_path())
		if display: display_results(info)
		else: return info
	else:
		pdat = polydat(name,img,px_size,scale_factor,'',True)
		skeleton = prepare(pdat)
		if display: analyze(skeleton,display)
		else: return analyze(skeleton,display)

# Displays the polydat. If the particle option is set, displays the indicated subparticle of a top-level polydat object.
def show(dat,particle=None,imtype='',pad=0,cmap='viridis'):
	if particle == None:
		if imtype == 'mask': img = load(_os.path.join(dat.get_id(),'mask')).get_img()
		elif imtype == 'skeleton': img = load(_os.path.join(dat.get_id(),'mask','skeleton')).get_img()
		elif imtype == 'points':
			end_points = load(_os.path.join(dat.get_id(),'end_points')).get_img()
			branch_points = load(_os.path.join(dat.get_id(),'branch_points')).get_img()
			bundle_points = load(_os.path.join(dat.get_id(),'bundle_points')).get_img()
			img = dat.get_img() + 3*end_points + branch_points + bundle_points
		else: img = dat.get_img()
		_plt.imshow(img,cmap=cmap)
		_plt.show()
	else:
		_plt.imshow(dat.get_particle(particle,imtype,pad),cmap=cmap)
		_plt.show()

# Returns a simple graph indicating the structure of the polydat. The input must have identified points.
def graphify(dat,particle=None):
	if particle == None: path = _os.path.join(dat.get_id(),'mask','skeleton')
	else: path = _os.path.join(dat.get_id(),'mask','skeleton',f'particle-{particle}')

	particles = split_particle(load(path))

	end_points = load(_os.path.join(path,'end_points')).get_img()
	branch_points = load(_os.path.join(path,'branch_points')).get_img()
	bundle_points = load(_os.path.join(path,'bundle_points')).get_img()

	branch_points = utilities.relu(branch_points-bundle_points)

	branch_coords = _np.array(_np.where(end_points + branch_points == 1)).T

	bundle_coords = []
	labeled_bundles = filters._label(bundle_points,connectivity=1)
	for v in range(_np.max(labeled_bundles)):
		bundle_coords.append(_np.array(_np.where(labeled_bundles == v+1)).T)

	lbranch = len(branch_coords)
	lbundle = len(bundle_coords)

	graph = dict([(i,[]) for i in range(lbranch + lbundle)])

	for i in range(lbranch):
		for p in range(len(particles)):
			if particles[p][branch_coords[i][0],branch_coords[i][1]] == 1:
				for j in range(lbranch):
					if i != j and particles[p][branch_coords[j][0],branch_coords[j][1]] == 1:
						if j not in graph[i]: graph[i].append(j)
				for j in range(lbranch, lbranch + lbundle):
					if (particles[p][bundle_coords[j-lbranch][...,0],bundle_coords[j-lbranch][...,1]] == 1).any():
						if j not in graph[i]: graph[i].append(j)
            
	for i in range(lbranch, lbranch + lbundle):
		for p in range(len(particles)):
			if (particles[p][bundle_coords[i-lbranch][...,0],bundle_coords[i-lbranch][...,1]] == 1).any():
				for j in range(lbranch):
					if particles[p][branch_coords[j][0],branch_coords[j][1]] == 1:
						if j not in graph[i]: graph[i].append(j)
				for j in range(lbranch, lbranch + lbundle):
					if i != j and (particles[p][bundle_coords[j-lbranch][...,0],bundle_coords[j-lbranch][...,1]] == 1).any():
						if j not in graph[i]: graph[i].append(j)

	return graph