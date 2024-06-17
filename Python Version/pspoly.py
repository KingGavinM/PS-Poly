# Basic Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from shutil import rmtree

# PSPoly Imports
import filters
import length
import utilities

pspoly_home = os.getcwd()
pspoly_save_option = 'load'
def initialize(home=pspoly_home,save_option=pspoly_save_option):
	global pspoly_home
	pspoly_home = home
	global pspoly_save_option
	pspoly_save_option = save_option

def load(address):
	try:
		info = open(os.path.join(pspoly_home,address,'info.txt')); lines = info.readlines(); info.close()
		pdat = polydat(lines[3][:-1],os.path.join(pspoly_home,address,'data.npy'),float(lines[7]),root=lines[5][:-1],save=False)
		pdat._date = lines[1][:-1]
		return pdat
	except:
		print(f'WARNING: A polydat object \'{address}\' could not be loaded.')

'''
The polydat type is the main datatype for PSPoly Python. The initializer takes the name of the polydat, its base image, and the pixel size
of the base image in nanometers as arguments. The scale factor is an optional argument which expands individual pixels for subpixel processing.
The expansion process will automatically adjust the saved pixel size so that anaylsis will remain accurate. The root argument gives the path
to the super folder of the polydat within the pspoly_home directory. For example, the data object ex at {pspoly_home}/CL_test/mask/skeleton/ex
would have root CL_test/mask/skeleton. No root should be specified for a polydat stored directly in the pspoly_home directory. The save option
defaults to True and specifies whether or not the polydat should be stored to disk. This option should usually not be changed.
'''
class polydat:
	def __init__(self,name,img,px_size,scale_factor=1,root='',save=True):

		self._name = name
		self._root = root

		self._px_size = px_size / scale_factor
		self._date = str(date.today())

		if type(img) == str:
			if os.path.splitext(img)[-1] == '.npy': temp_img = np.load(img)
			else: temp_img = utilities.to_gray(plt.imread(img))
		else: temp_img = img

		if scale_factor != 1:
			self._img = np.zeros((scale_factor*temp_img.shape[0],scale_factor*temp_img.shape[1]))
			for iy in range(self._img.shape[0]):
				for ix in range(self._img.shape[1]):
					self._img[iy,ix] = temp_img[iy//scale_factor,ix//scale_factor]
		else: self._img = temp_img

		if save: self.save_dat()


	# Saves the polydat object to disk. Runs if the save option is set to True.
	def save_dat(self):
		if not os.path.exists(self.get_path()):
			os.mkdir(self.get_path())
			info = open(self.get_infpath(),'w'); info.write(f'Creation Date\n{self.get_date()}\nName\n{self.get_name()}\nRoot\n{self.get_root()}\nPixel Size\n{self.get_px_size()}'); info.close()
			np.save(self.get_datpath(),self.get_img())
		elif pspoly_save_option == 'ask':
			option = input(f'WARNING: A polydat object \'{self.get_id()}\' already exists.\nWould you like to overwrite it? (y/n): ')
			if option == 'y':
				print(f'An old polydat object \'{self.get_id()}\' has been overwritten.')
				rmtree(self.get_path())
				os.mkdir(self.get_path())
				info = open(self.get_infpath(),'w'); info.write(f'Creation Date\n{self.get_date()}\nName\n{self.get_name()}\nRoot\n{self.get_root()}\nPixel Size\n{self.get_px_size()}'); info.close()
				np.save(self.get_datpath(),self.get_img())
			elif option == 'n':
				print(f'An old polydat object \'{self.get_id()}\' has been loaded.')
				self = load(self.get_id())
			else:
				print(f'WARNING: An invalid option has been selected. Loading an old polydat object \'{self.get_id()}\'.')
				self = load(self.get_id())
		elif pspoly_save_option == 'overwrite':
			print(f'WARNING: An old polydat object \'{self.get_id()}\' has been overwritten.')
			rmtree(self.get_path())
			os.mkdir(self.get_path())
			info = open(self.get_infpath(),'w'); info.write(f'Creation Date\n{self.get_date()}\nName\n{self.get_name()}\nRoot\n{self.get_root()}\nPixel Size\n{self.get_px_size()}'); info.close()
			np.save(self.get_datpath(),self.get_img())
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
		return np.copy(self._img)

	def get_px_size(self):
		return self._px_size

	def get_date(self):
		return self._date

	# Returns the path to the polydat.
	def get_path(self):
		if self._root == '':
			return  os.path.join(pspoly_home,self._name)
		else:
			return os.path.join(pspoly_home,self._root,self._name)

	# Returns the path containing the polydat.
	def get_superpath(self):
		if self._root == '':
			return pspoly_home
		else:
			return os.path.join(pspoly_home,self._root)

	# Returns the path to the data.npy file of the polydat.
	def get_datpath(self):
		if self._root == '':
			return os.path.join(pspoly_home,self._name,'data.npy')
		else:
			return os.path.join(pspoly_home,self._root,self._name,'data.npy')

	# Returns the path to the info.txt file of the polydat.
	def get_infpath(self):
		if self._root == '':
			return os.path.join(pspoly_home,self._name,'info.txt')
		else:
			return os.path.join(pspoly_home,self._root,self._name,'info.txt')

	# Returns the path to the bounds.txt file of the polydat.
	def get_bndpath(self):
		if self._root == '':
			return os.path.join(pspoly_home,self._name,'bounds.txt')
		else:
			return os.path.join(pspoly_home,self._root,self._name,'bounds.txt')

	# Returns the path to the verify.txt file of the polydat.
	def get_vpath(self):
		if self._root == '':
			return os.path.join(pspoly_home,self._name,'verify.txt')
		else:
			return os.path.join(pspoly_home,self._root,self._name,'verify.txt')

	# Returns the name of the top-level polydat object in which the polydat is nested.
	def get_oname(self):
		path = self.get_path()

		while True:
			ht = os.path.split(path)
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
			return os.path.join(self._root,self._name)

	# Creates a polydat object nested within the polydat.
	def subdat(self,name,img,px_size,scale_factor=1,save=True):
		if self._root == '':
			root = self._name
		else:
			root = os.path.join(self._root,self._name)
		return polydat(name,img,px_size,scale_factor,root,save)
		
	# Lists all folders within the polydat. The folder names are equivalent to the names of the nested polydat objects, unless a subfolder has been created manually.
	def get_subdata(self):
		path = self.get_path()
		ldir = os.listdir(path)
		return [f for f in ldir if os.path.isdir(os.path.join(path,f))]

	def isolate_skeleton(self,particle):
		return load(os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))

	def get_particle(self,particle,skeleton=False,pad=0):
		if skeleton:
			skel = load(os.path.join(self.get_id(),'mask','skeleton'))
			shp = self.get_img().shape
			pdat = load(os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))
			bnds = pdat.get_bnd()
			return skel.get_img()[max(0,bnds[0][0]-pad):min(bnds[0][1]+1+pad,shp[0]),max(0,bnds[1][0]-pad):min(bnds[1][1]+1+pad,shp[1])]
		else:
			shp = self.get_img().shape
			pdat = load(os.path.join(self.get_id(),'mask','skeleton',f'particle-{particle}'))
			bnds = pdat.get_bnd()
			return self.get_img()[max(0,bnds[0][0]-pad):min(bnds[0][1]+1+pad,shp[0]),max(0,bnds[1][0]-pad):min(bnds[1][1]+1+pad,shp[1])]

# Returns pspoly_home.
def get_home():
	return pspoly_home

# Returns pspoly_load_option.
def get_save_option():
	return pspoly_save_option

'''
The threshold method creates a subdat consisting of a thresholded mask of the original polydat. The default options will apply the OTSU method
to the polydat data, which automatically determines the appropriate threshold height. All regions in contact with the border of the image will
be removed from the mask to improve the quality of analysis. The threshold argument specifies the method for creating a mask image from the
original. If a method is used that does not automatically determine the threshold height, the height to be used should be passed to the t
argument. The save option specifies whether or not the subdat will be saved to disk.
'''
def threshold(dat,threshold_function=filters.otsu_central,t=None,save=True):
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
	subdata = [d for d in dat.get_subdata() if d[:8] == 'particle']
	return [load(os.path.join(dat.get_id(),d)) for d in subdata]

'''
Creates four subdat objects: one for the end points, one for the branch points, one for the bundle points, and one for the secondary branch
points in a polydat. The original data must be in skeletonized form. An end point is any location in a skeleton where there is only
connectivity in one direction. A branch point has connectivity in more than two directions. A bundle point is a group of four adjacent active
points in a skeleton, which would ideally be classified as branch points, but they are not because the different branches do not all come from
the same location. Secondary branch points are those points in a skeleton that would be classified as branch points if the original branch
points were removed.
'''
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

	end_points = load(os.path.join(path,'end_points')).get_img()
	branch_points = load(os.path.join(path,'branch_points')).get_img()
	bundle_points = load(os.path.join(path,'bundle_points')).get_img()
	branch_points2 = load(os.path.join(path,'branch_points2')).get_img()

	labeled_img = filters.label(utilities.relu(img-bundle_points-branch_points-branch_points2))

	particles = []
	for v in range(np.max(labeled_img)):
		particles.append((labeled_img == v+1).astype('int'))

	bc = np.array(np.where(branch_points == 1)).T
	b2c = np.array(np.where(branch_points2 == 1)).T
	bdc = np.array(np.where(bundle_points == 1)).T

	# This is allowing multiple areas to be added back at once
	for particle in particles:
		particlep = np.pad(particle,1)
		particlepc = np.copy(particlep)
		for i in range(b2c.shape[0]):
			if np.sum(particlep[b2c[i,0]:b2c[i,0]+3,b2c[i,1]:b2c[i,1]+3]) > 0:
				particle[b2c[i,0],b2c[i,1]] = particlepc[b2c[i,0]+1,b2c[i,1]+1] = 1
		particlep = np.copy(particlepc)
		for i in range(bc.shape[0]):
			if np.sum(particlep[bc[i,0]:bc[i,0]+3,bc[i,1]:bc[i,1]+3]) > 0:
				particle[bc[i,0],bc[i,1]] = 1
		for i in range(bdc.shape[0]):
			if np.sum(particlep[bdc[i,0]:bdc[i,0]+3,bdc[i,1]:bdc[i,1]+3]) > 0:
				particle[bdc[i,0],bdc[i,1]] = 1

	composite = sum(particles)
	labeled_missed = filters.label(utilities.relu(img-composite))

	particles_missed = []
	for v in range(np.max(labeled_missed)):
		particles_missed.append((labeled_missed == v+1).astype('int'))

	for particle in particles_missed:
		particlep = np.pad(particle,1)
		for i in range(bc.shape[0]):
			if np.sum(particlep[bc[i,0]:bc[i,0]+3,bc[i,1]:bc[i,1]+3]) > 0:
				particle[bc[i,0],bc[i,1]] = 1
		for i in range(bdc.shape[0]):
			if np.sum(particlep[bdc[i,0]:bdc[i,0]+3,bdc[i,1]:bdc[i,1]+3]) > 0:
				particle[bdc[i,0],bdc[i,1]] = 1

	particles_additional = []
	for particle in particles_missed:
		n_e = np.sum(end_points*particle)
		n_b = np.max(filters.label((branch_points+bundle_points)*particle > 0).astype('int'))
		if n_e+n_b > 1:
			particles_additional.append(particle)

	particles.extend(particles_additional)

	return particles

# Returns a simple graph indicating the structure of the polydat. The input must have identified points.
def graphify(dat):
	particles = split_particle(dat)
	path = dat.get_id()
	bundle_points = load(os.path.join(path,'bundle_points')).get_img()
	end_points = load(os.path.join(path,'end_points')).get_img()
	branch_points = load(os.path.join(path,'branch_points')).get_img()

	bundle_coords = []
	labeled_bundles = filters.label(bundle_points)
	for v in range(np.max(labeled_bundles)):
		bundle_coords.append(np.array(np.where(labeled_bundles == v+1)).T)

	branch_coords = np.array(np.where(end_points + branch_points == 1)).T

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

	branch_groups = []
	labeled_branches = filters.label(end_points + branch_points)
	for v in range(np.max(labeled_branches)):
		branch_groups.append(np.array(np.where(labeled_branches == v+1)).T)

	lgroups = len(branch_groups)

	condensed_graph = dict([(i,[]) for i in range(lgroups + lbundle)])
	mapping = np.zeros(lbranch + lbundle,dtype='int')

	for c in range(lbranch):
		for g in range(lgroups):
			for coord in branch_groups[g]:
				if (branch_coords[c] == coord).all():
					mapping[c] = g

	for c in range(lbundle):
		mapping[lbranch+c] = lgroups+c

	for n0 in graph:
		for n1 in graph[n0]:
			if mapping[n1] not in condensed_graph[mapping[n0]] and mapping[n1] != mapping[n0]: condensed_graph[mapping[n0]].append(mapping[n1])

	return condensed_graph


# Completes all necessary steps prior to analyzing the polydat. Returns a skeletonized polydat, which will be the input to the analyze method. Will only return the skeleton if the polydat has been verified.
def prepare(dat):
	if not os.path.isfile(dat.get_vpath()):
		mask = threshold(dat)
		skeleton = skeletonize(mask)
		particles = separate_particles(skeleton)
		for particle in particles: identify_points(particle)
		dat.verify()
		return skeleton
	else:
		print(f'WARNING: An old polydat object has been loaded instead of preparing \'{dat.get_id()}\' again.')
		if dat.get_root() == '':
			return load(os.path.join(dat.get_name(),'mask/skeleton'))
		else:
			return load(os.path.join(dat.get_root(),dat.get_name(),'mask/skeleton'))
		
# Displays the polydat. If the particle option is set, displays the indicated subparticle of a top-level polydat object.
def show(dat,particle=None,skeleton=False,pad=0):
	if particle == None:
		plt.imshow(dat.get_img())
		plt.show()
	else:
		plt.imshow(dat.get_particle(particle,skeleton,pad))
		plt.show()

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
	end_points = load(os.path.join(path,'end_points')).get_img()
	branch_points = load(os.path.join(path,'branch_points')).get_img()
	bundle_points = load(os.path.join(path,'bundle_points')).get_img()

	if threshold == None: threshold = 1.5*mean_height(load(dat.get_root()))

	skeleton = dat.get_img()
	height = measure_height(dat)
	high = skeleton*(height > threshold)

	nodes = np.max(filters.label(end_points)) + np.max(filters.label(branch_points)) + np.max(filters.label(bundle_points))

	if np.sum(high) == 0:
		if nodes == 0: return 'looped'
		else:
			particles = split_particle(dat)
			edges = len(particles)
			if nodes-edges == 1:
				if nodes == 2: return 'linear'
				else: return 'branched'
			else: return 'branched and looped'
	elif np.sum(high) / np.sum(skeleton) > noise: return 'noise particle'
	elif np.sum(high*(branch_points + bundle_points)) > 0: return 'overlapped'
	else:
		n = np.max(filters.label(high))
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

	print(f'Particle Type\t\tNumber of Features\t\tAverage length (nm)\t\tPercentage of Total particleization Length\nLinear\t\t\t\t{stats[0][0]}\t\t\t\t{stats[0][1]}\t\t\t\t{stats[0][2]}\nLooped\t\t\t\t{stats[1][0]}\t\t\t\t{stats[1][1]}\t\t\t\t{stats[1][2]}\nBranched (no looping)\t\t{stats[2][0]}\t\t\t\t{stats[2][1]}\t\t\t\t{stats[2][2]}\nBranched (with looping)\t\t{stats[3][0]}\t\t\t\t{stats[3][1]}\t\t\t\t{stats[3][2]}\nOverlapped\t\t\t{stats[4][0]}\t\t\t\t{stats[4][1]}\t\t\t\t{stats[4][2]}\n')
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

# Runs the full process on the polydat.
def run(img,px_size,scale_factor=1,name=None,display=True):
	if name == None:
		pdat = polydat('pspoly_temp',img,px_size,scale_factor,'',True)
		skeleton = prepare(pdat)
		info = analyze(skeleton,False)
		rmtree(pdat.get_path())
		if display: display_results(info)
		else: return info
	else:
		pdat = polydat(name,img,px_size,scale_factor,'',True)
		skeleton = prepare(pdat)
		if display: analyze(skeleton,display)
		else: return analyze(skeleton,display)