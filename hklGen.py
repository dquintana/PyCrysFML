# hklGen.py
# Generates predicted diffraction peak positions for a given unit cell and space
#   group, using the Fortran CFML library.
# Created 6/4/2013
# Last edited 6/7/2013

from math import floor, sqrt, log, tan, radians
from string import join, rstrip
from ctypes import cdll, Structure, c_int, c_float, c_char, c_bool, c_char_p, \
                   c_void_p, c_ulong, addressof, sizeof, POINTER

import numpy as np
import pylab

import lattice_calculator_procedural2 as latcalc

# This should be the location of the CFML library
lib = cdll["./libcrysfml.so"]

# SymmetryOp Attributes:
#   rot     - rotational part of symmetry operator (3 by 3 matrix)
#   trans   - translational part of symmetry operator (vector)
class SymmetryOp(Structure):
    _fields_ = [("rot", c_int*9), ("trans", c_float*3)]

# WyckoffPos Attributes:
#   multip      - multiplicity
#   site        - site symmetry
#   numElements - number of elements in orbit
#   origin      - origin
#   orbit       - strings containing orbit information
class WyckoffPos(Structure):
    _fields_ = [("multip", c_int), ("site", c_char*6),
                ("numElements", c_int), ("origin", c_char*40),
                ("orbit", c_char*40*48)]

# Wyckoff Attributes:
#   numOrbits   - number of orbits
#   orbits      - list of Wyckoff position objects
class Wyckoff(Structure):
    _fields_ = [("numOrbits", c_int), ("orbits", WyckoffPos*26)]

# SpaceGroup Attributes:
#   number          - space group number
#   symbol          - Hermann-Mauguin symbol
#   hallSymbol      - Hall symbol
#   xtalSystem      - Crystal system
#   laue            - Laue class
#   pointGroup      - corresponding point group
#   info            - additional information
#   setting         - space group setting information (IT, KO, ML, ZA, Table,
#                     Standard, or UnConventional)
#   hex             - true if space group is hexagonal
#   lattice         - lattice type
#   latticeSymbol   - lattice type symbol
#   latticeNum      - number of lattice points in a cell
#   latticeTrans    - lattice translations
#   bravais         - Bravais symbol and translations
#   centerInfo      - information about symmetry center
#   centerType      - 0 = centric (-1 not at origin), 1 = acentric,
#                     2 = centric (-1 at origin)
#   centerCoords    - fractional coordinates of inversion center
#   numOps          - number of symmetry operators in the reduced set
#   multip          - multiplicity of the general position
#   numGens         - minimum number of operators to generate the group
#   symmetryOps     - list of symmetry operators
#   symmetryOpsSymb - string form of symmetry operator objects
#   wyckoff         - object containing Wyckoff information
#   asymmetricUnit  - direct space parameters for the asymmetric unit
class SpaceGroup(Structure):
    _fields_ = [("number", c_int), ("symbol", c_char*20),
                ("hallSymbol", c_char*16), ("xtalSystem", c_char*12),
                ("laue", c_char*5), ("pointGroup", c_char*5),
                ("info", c_char*5), ("setting", c_char*80),
                ("hex", c_int), ("lattice", c_char),
                ("latticeSymbol", c_char*2), ("latticeNum", c_int),
                ("latticeTrans", c_float*3*12), ("bravais", c_char*51),
                ("centerInfo", c_char*80), ("centerType", c_int),
                ("centerCoords", c_float*3), ("numOps", c_int),
                ("multip", c_int), ("numGens", c_int),
                ("symmetryOps", SymmetryOp*192),
                ("symmetryOpsSymb", c_char*40*192),
                ("wyckoff", Wyckoff), ("asymmetricUnit", c_float*6)]

# CrystalCell Attributes:
#   length, angle           - arrays of unit cell parameters
#   lengthSD, angleSD       - standard deviations of parameters
#   rLength, rAngle         - arrays of reciprocal cell parameters
#   GD, GR                  - direct and reciprocal space metric tensors
#   xtalToOrth, orthToXtal  - matrices to convert between orthonormal and
#                             crystallographic bases
#   BLB, invBLB             - Busing-Levy B-matrix and its inverse
#   volume, rVolume         - direct and reciprocal cell volumes
#   cartType                - Cartesian reference frame type (cartType = 'A'
#                             designates x || a)
class CrystalCell(Structure):
    _fields_ = [("length", c_float*3), ("angle", c_float*3),
                ("lengthSD", c_float*3), ("angleSD", c_float*3), 
                ("rLength", c_float*3), ("rAngle", c_float*3),
                ("GD", c_float*9), ("GR", c_float*9),
                ("xtalToOrth", c_float*9), ("orthToXtal", c_float*9),
                ("BLB", c_float*9), ("invBLB", c_float*9),
                ("volume", c_float), ("rVolume", c_float),
                ("cartType", c_char)]

# Reflection Attributes:
#   hkl     - list containing hkl indices for the reflection
#   multip  - multiplicity
#   s       - s = sin(theta)/lambda = 1/(2d) [No 4*pi factor!]
class Reflection(Structure):
    _fields_ = [("hkl", c_int*3), ("multip", c_int), ("s", c_float)]

# Gaussian: represents a Gaussian function that can be evaluated at any
#   2*theta value. u, v, and w are fitting parameters.
class Gaussian(object):
    def __init__(self, center, u, v, w, I, hkl=[0,0,0]):
        self.C0 = 4*log(2)
        self.center = center    # 2*theta position
        self.u = u
        self.v = v
        self.w = w
        self.I = I
        try:
            self.H = sqrt(u*(tan(radians(center/2))**2)
                          + v*tan(radians(center/2)) + w)
        except ValueError:
            self.H = 0
        self.scale = self.I * sqrt(self.C0/np.pi)/self.H
        self.hkl = hkl

    # __call__: returns the value of the Gaussian at some 2*theta positions
    def __call__(self, x):
        return self.scale * np.exp(-self.C0*(x-self.center)**2/self.H**2)

    def add(self, v, x):
        idx = (x>self.center-self.H*3) & (x<self.center+self.H*3)
        v[idx] += self.__call__(x[idx])

# LinSpline: represents a linear spline function to be used for the background
class LinSpline(object):
    def __init__(self, arg1, arg2=None):
        if type(arg1) == type(np.array([])):
            # read in x and y coordinates from lists
            self.x = np.copy(arg1)
            self.y = np.copy(arg2)
        elif type(arg1) == str:
            # read in x and y coordinates from a file
            self.x, self.y = np.loadtxt(arg1, dtype=float, skiprows=5, unpack=True)

    # __call__: returns the interpolated y value at some x position
    def __call__(self, x):
        # locate the two points to interpolate between
        return np.interp(x, self.x, self.y)
        
    def __repr__(self):
        return "LinSpline(" + str(self.x) + ", " + str(self.y) + ")"

# DVDim: contains array dimension information. Attributes:
#   stride_mult - stride multiplier for the dimension
#   lower_bound - first index for the dimension
#   upper_bound - last index for the dimension
class DVDim(Structure):
    _fields_ = [("stride_mult", c_ulong), ("lower_bound", c_ulong),
                ("upper_bound", c_ulong)]

# DV: dope vector for gfortran that passes array information. Attributes:
#   base_addr   - base address for the array
#   base        - base offset
#   dtype       - contains the element size, type (3 bits), and rank (3 bits)
#   dim         - DVDim object for the vector
class DV(Structure):
    _fields_ = [("base_addr",c_void_p), ("base", c_void_p), ("dtype", c_ulong),
                ("dim", DVDim*7)]

# dv_dtype: calculates the "dtype" attribute for a dope vector
def dv_dtype(size,type,rank): return size*64+type*8+rank

# build_struct_dv: constructs a dope vector for an array of derived types
def build_struct_dv(array):    
    dv = DV()
    dv.base_addr = addressof(array)
    dv.base = c_void_p()
    dv.dtype = dv_dtype(sizeof(array[0]), 5, 1) # 5 = derived type
    dv.dim[0].stride_mult = 1
    dv.dim[0].lower_bound = 1
    dv.dim[0].upper_bound = len(array)
    return dv

# twoTheta: converts a sin(theta)/lambda position to a 2*theta position
def twoTheta(s, wavelength):
#    if (s*wavelength >= 1): return 180.0
#    if (s*wavelength <= 0): return 0.0
    return 2*np.degrees(np.arcsin(s*wavelength))
    
def getS(tt, wavelength):
    return np.sin(np.radians(tt/2))/wavelength

# setSpaceGroup: constructs a SpaceGroup object from a provided symbol/number
def setSpaceGroup(name, spaceGroup):
    #print >>sys.stderr, name, spaceGroup
    fn = lib.__cfml_crystallographic_symmetry_MOD_set_spacegroup
    fn.argtypes = [c_char_p, POINTER(SpaceGroup), c_void_p, POINTER(c_int),
                   c_char_p, c_char_p, c_int, c_int, c_int]
    fn.restype = None
    fn(name, spaceGroup, None, None, None, None, len(name), 0, 0)
    return

# setCrystalCell: constructs a CrystalCell object from provided parameters
def setCrystalCell(length, angle, cell):
    fn = lib.__cfml_crystal_metrics_MOD_set_crystal_cell
    float3 = c_float*3
    fn.argtypes = [POINTER(float3), POINTER(float3), POINTER(CrystalCell),
                   POINTER(c_char), POINTER(float3)]
    fn.restype = None
    fn(float3(*length), float3(*angle), cell, None, None)
    return

# getMaxNumRef: returns the maximum number of reflections for a given cell
def getMaxNumRef(sMax, volume, sMin=0.0, multip=2):
    fn = lib.__cfml_reflections_utilities_MOD_get_maxnumref
    fn.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float),
                   POINTER(c_int)]
    fn.restype = c_int
    numref = fn(c_float(sMax), c_float(volume), c_float(sMin), c_int(multip))
    return numref

# hklUni: constructs a list of unique reflections in a specified range
#   If code == "r", then d-spacings are used
def hklUni(cell, spaceGroup, sMin, sMax, code, numRef, maxRef):
    fn = lib.__cfml_reflections_utilities_MOD_hkl_uni_reflect
    c_ReflectionArray = Reflection*max(maxRef,1)
    reflections = c_ReflectionArray()
    fn.argtypes = [POINTER(CrystalCell), POINTER(SpaceGroup), POINTER(c_bool),
                   POINTER(c_float), POINTER(c_float), c_char_p,
                   POINTER(c_int), POINTER(DV), POINTER(c_bool)]
    fn.restype = None
    fn(cell, spaceGroup, c_bool(True), c_float(sMin), c_float(sMax),
       code, numRef, build_struct_dv(reflections), c_bool(False))
    return reflections

# inputInfo: requests space group and cell information, wavelength, and range
#   of interest
def inputInfo():
    # Input the space group name and create it
    groupName = raw_input("Enter space group "
                          "(HM symbol, Hall symbol, or number): ")
    spaceGroup = SpaceGroup()
    setSpaceGroup(groupName, spaceGroup)
    
    # Remove excess spaces from Fortran
    spaceGroup.xtalSystem = rstrip(spaceGroup.xtalSystem)
    length = [0, 0, 0]
    angle = [0, 0, 0]
    
    # Ask for parameters according to the crystal system and create the cell
    if (spaceGroup.xtalSystem == "Triclinic"):
        length[0], length[1], length[2], angle[0], angle[1], angle[2] = \
            input("Enter cell parameters (a, b, c, alpha, beta, gamma): ")
    elif (spaceGroup.xtalSystem == "Monoclinic"):
        angle[0] = angle[2] = 90
        length[0], length[1], length[2], angle[1] = \
            input("Enter cell parameters (a, b, c, beta): ")
    elif (spaceGroup.xtalSystem == "Orthorhombic"):
        angle[0] = angle[1] = angle[2] = 90
        length[0], length[1], length[2] = \
            input("Enter cell parameters (a, b, c): ")
    elif (spaceGroup.xtalSystem == "Tetragonal"):
        angle[0] = angle[1] = angle[2] = 90
        length[0], length[2] = input("Enter cell parameters (a, c): ")
        length[1] = length[0]
    elif (spaceGroup.xtalSystem in ["Rhombohedral", "Hexagonal"]):
        angle[0] = angle[1] = 90
        angle[2] = 120
        length[0], length[2] = input("Enter cell parameters (a, c): ")
        length[1] = length[0]
    elif (spaceGroup.xtalSystem == "Cubic"):
        angle[0] = angle[1] = angle[2] = 90
        length[0] = input("Enter cell parameter (a): ")
        length[1] = length[2] = length[0]
    cell = CrystalCell()
    setCrystalCell(length, angle, cell)
    
    # Input the wavelength and range for hkl calculation (and adjust the range
    #   if necessary)
    wavelength = input("Enter the wavelength: ")
    sMin, sMax = input("Enter the sin(theta)/lambda interval: ")
    adjusted = False    
    if (sMin < 0):
        sMin = 0
        adjusted = True
    if (sMax > 1.0/wavelength):
        sMax = 1.0/wavelength
        adjusted = True
    if (adjusted):
        print "sin(theta)/lambda interval adjusted to [%f, %f]" % (sMin, sMax)
    return (spaceGroup, cell, wavelength, sMin, sMax)
    
# hklGen: generates a list of reflections
def hklGen(spaceGroup, cell, wavelength, sMin, sMax):
    # Calculate the reflection positions
    maxReflections = getMaxNumRef(sMax+0.2, cell.volume,
                                  multip=spaceGroup.multip)
    # Create a reference that will be modified by hklUni()
    reflectionCount = c_int()
    reflections = hklUni(cell, spaceGroup, sMin, sMax, 's', reflectionCount,
                         maxReflections)
    reflections = reflections[:reflectionCount.value]
    return reflections

# printReflections: outputs spacegroup information and a list of reflections
def printReflections(reflections, spaceGroup, wavelength, sMin, sMax):
    reflectionCount = len(reflections)
    print
    print "Space group information"
    print "---------------------------------------"
    print "              Number: ", spaceGroup.number
    print "          H-M Symbol: ", spaceGroup.symbol
    print "         Hall Symbol: ", spaceGroup.hallSymbol
    print "      Crystal System: ", spaceGroup.xtalSystem
    print "          Laue Class: ", spaceGroup.laue
    print "         Point Group: ", spaceGroup.pointGroup
    print "General Multiplicity: ", spaceGroup.multip
    print "---------------------------------------"
    print
    print "%d reflections found for %f < sin(theta)/lambda < %f" \
            % (reflectionCount, sMin, sMax)
    print "                        (%f < 2*theta < %f)" % \
            (twoTheta(sMin, wavelength), twoTheta(sMax, wavelength))
    print
    print " h k l  mult  sin(theta)/lambda  2*theta"
    for refl in reflections:
        print "(%d %d %d)  %d        %f       %f" % \
                (refl.hkl[0], refl.hkl[1], refl.hkl[2], refl.multip, refl.s,
                 twoTheta(refl.s, wavelength))
    print
    return

# makeGaussians() creates a series of Gaussians to represent the powder
#   diffractionn pattern
def makeGaussians(reflections, coeffs, I, wavelength):
    gaussians = [Gaussian(twoTheta(rk.s, wavelength),
                          coeffs[0], coeffs[1], coeffs[2], Ik, rk.hkl)
                 for rk,Ik in zip(reflections,I)]
    return gaussians

# getIntensity: calculates the intensity at a given 2*theta position, or for an
#   array of 2*theta positions
def getIntensity(gaussians, background, tt):
    #return background(tt) + sum(g(tt) for g in gaussians)
    v = background(tt)
    for g in gaussians:
        g.add(v,tt)
    return v

# plotPattern: given a series of Gaussians and a background, plots the predicted
#   intensity at every 2*theta position in a specified range, as well as the
#   observed intensity everywhere on a given list of points
def plotPattern(gaussians, background, ttObs, observed, ttMin, ttMax, ttStep,
                exclusions=None):
    numPoints = int(floor((ttMax-ttMin)/ttStep)) + 1
    ttCalc = np.linspace(ttMin, ttMax, numPoints)
    if(exclusions != None): ttCalc = removeRange(ttCalc, exclusions)
    intensity = np.array(getIntensity(gaussians, background, ttCalc))
    pylab.plot(ttCalc, intensity, '-', label="Caclulated")
    pylab.plot(ttObs, observed, '-', label="Observed")
    pylab.xlabel(r"$2 \theta$")
    pylab.ylabel("Intensity")
    pylab.legend()
    for g in gaussians:
        pylab.text(g.center, np.interp(g.center, ttCalc, intensity),
                   " " + "".join([str(g.hkl[0]), str(g.hkl[1]), str(g.hkl[2])]),
                   ha="center", va="bottom", rotation="vertical")
    # TODO: label peaks with hkl indices
    return

# removeRange: takes in an array of 2*theta intervals and removes them from
#   consideration for data analysis, with an optional argument for removing the
#   corresponding intensities as well
def removeRange(tt, remove, intensity=None):
    if (remove == None):
        if (intensity != None): return (tt, intensity)
        else: return tt
    if (type(remove[0]) not in [list, type(np.array([]))]):
        # single interval
        keepEntries = (tt < remove[0]) | (tt > remove[1])
        tt = tt[keepEntries]
        if (intensity != None):
            intensity = intensity[keepEntries]
            return (tt, intensity)
        else: return tt
    else:
        # array of intervals
        if (intensity != None):
            for interval in remove:
                tt, intensity = removeRange(tt, interval, intensity)
            return (tt, intensity)
        else:
            for interval in remove:
                tt  = removeRange(tt, interval)
            return tt

# chiSquare: returns the chi-square statistic between observed and generated
#   data
#def chiSquare(observed, gaussians, background, tt):
#    expected = np.array(getIntensity(gaussians, background, tt))
#    return sum((observed-expected)**2/expected)

# Model: represents an object that can be used with bumps for optimization
#   purposes
class Model(object):

    def __init__(self, tt, observed, background, u, v, w, I,
                 wavelength, spaceGroupName, cell):
        #print >>sys.stderr, "create spacegroup"
        self.spaceGroup = SpaceGroup()
        #print >>sys.stderr, "created spacegroup"
        setSpaceGroup(spaceGroupName, self.spaceGroup)

        #print >>sys.stderr, "set spacegroup"
        self.tt = tt
        self.observed = observed
        self.background = background        
        self.u = Parameter(u, name='u')
        self.v = Parameter(v, name='v')
        self.w = Parameter(w, name='w')
        self.wavelength = wavelength
        self.cell = cell
        self.ttMin = min(self.tt)
        self.ttMax = max(self.tt)
        self.sMin = getS(self.ttMin, self.wavelength)
        self.sMax = getS(self.ttMax, self.wavelength)
        
        maxCell = CrystalCell()
        # TODO: make this work for other crystal cell types
        setCrystalCell([cell.a.bounds.limits[1], cell.a.bounds.limits[1],
                        cell.c.bounds.limits[1]], [90,90,120], maxCell)
        self.maxReflections = hklGen(self.spaceGroup, maxCell,
                                     self.wavelength, self.sMin, self.sMax)
        self.reflections = np.copy(self.maxReflections)
        self.I = [Parameter(I, name='I[%d]'%k) for k,_ in enumerate(self.reflections)]
        #print >>sys.stderr, "updating"
        self.update()
        #print >>sys.stderr, "updated"

    def parameters(self):
        return {'u': self.u,
                'v': self.v,
                'w': self.w,
                'I': self.I,
                'cell': self.cell.parameters(),
                }
        
    def numpoints(self):
        return len(self.observed)

    def theory(self):
        return getIntensity(self.gaussians, self.background, self.tt)

    def residuals(self):
        return (self.theory() - self.observed)/np.sqrt(self.observed)
        
    def nllf(self):
        return np.sum(self.residuals()**2)
        #return chiSquare(self.observed, self.gaussians, self.background,
        #                 self.tt)
                         
    def plot(self, view="linear"):
        plotPattern(self.gaussians, self.background, self.tt, self.observed,
                    self.ttMin, self.ttMax,0.01)
                    
    def update(self):
        self.cell.update()
        lattice = latcalc.Lattice(self.cell.a.value, self.cell.a.value,
                                  self.cell.c.value, 90,90,120)
        h, k, l = np.array(zip(*[reflection.hkl for reflection in self.maxReflections]))
        ttPos = lattice.calc_twotheta(self.wavelength, h, k, l)
        #print ttPos
        #include = [False]*len(self.reflections)
        for i in xrange(len(self.reflections)):
            self.reflections[i].s = getS(ttPos[i], self.wavelength)
            #if (self.sMin < self.reflections[i].s < self.sMax): include[i] = True
        self.gaussians = makeGaussians(self.reflections,
                                       [self.u.value, self.v.value, self.w.value],
                                       [Ik.value for Ik in self.I], self.wavelength)

# Triclinic/Monoclinic/Orthorhombic/Tetragonal/Hexagonal/CubicCell: classes
#   that contain lattice information with refinable parameters to interface
#   with bumps
class TriclinicCell(object):
    def __init__(self, a, b, c, alpha, beta, gamma):
        self.cell = CrystalCell()
        self.a = Parameter(a, name='a')
        self.b = Parameter(b, name='b')
        self.c = Parameter(c, name='c')
        self.alpha = Parameter(alpha, name='alpha')
        self.beta = Parameter(beta, name='beta')
        self.gamma = Parameter(gamma, name='gamma')
        self.update()
    def parameters(self):
        return {'a': self.a, 'b': self.b, 'c': self.c,
                'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}
    def update(self):
        a = self.a.value
        b = self.b.value
        c = self.c.value
        alpha = self.alpha.value
        beta = self.beta.value
        gamma = self.gamma.value        
        setCrystalCell([a,b,c], [alpha, beta, gamma], self.cell)

class MonoclinicCell(object):
    def __init__(self, a, b, c, beta):
        self.cell = CrystalCell()
        self.a = Parameter(a, name='a')
        self.b = Parameter(b, name='b')
        self.c = Parameter(c, name='c')
        self.beta = Parameter(beta, name='beta')
        self.update()
    def parameters(self):
        return {'a': self.a, 'b': self.b, 'c': self.c,
                'beta': self.beta}
    def update(self):
        a = self.a.value
        b = self.b.value
        c = self.c.value
        beta = self.beta.value       
        setCrystalCell([a,b,c], [90, beta, 90], self.cell)

class OrthorhombicCell(object):
    def __init__(self, a, b, c):
        self.cell = CrystalCell()
        self.a = Parameter(a, name='a')
        self.b = Parameter(b, name='b')
        self.c = Parameter(c, name='c')
        self.update()
    def parameters(self):
        return {'a': self.a, 'b': self.b, 'c': self.c}
    def update(self):
        a = self.a.value
        b = self.b.value
        c = self.c.value       
        setCrystalCell([a,b,c], [90,90,90], self.cell)

class TetragonalCell(object):
    def __init__(self, a, c):
        self.cell = CrystalCell()
        self.a = Parameter(a, name='a')
        self.c = Parameter(c, name='c')
        self.update()
    def parameters(self):
        return {'a': self.a, 'c': self.c}
    def update(self):
        a = self.a.value
        c = self.c.value     
        setCrystalCell([a,a,c], [90,90,90], self.cell)

class HexagonalCell(object):
    def __init__(self, a, c):
        self.cell = CrystalCell()
        self.a = Parameter(a, name='a')
        self.c = Parameter(c, name='c')
        self.update()
    def parameters(self):
        return {'a': self.a, 'c': self.c}
    def update(self):
        a = self.a.value
        c = self.c.value       
        setCrystalCell([a,a,c], [90,90,120], self.cell)
    def maxvolume(self):
        amax = self.a.bounds.limits[1]
        bmax = amax
        cmax = self.c.bounds.limits[1]        
        return amax, bmax, cmax, 90, 90, 120
     #import sys; print >>sys.stderr, cell.c.bounds.limits
 
class CubicCell(object):
    def __init__(self, a):
        self.cell = CrystalCell()
        self.a = Parameter(a, name='a')
        self.update()
    def parameters(self):
        return {'a': self.a}
    def update(self):
        a = self.a.value
        setCrystalCell([a,a,a], [90,90,90], self.cell)

def testInfo():
    length = [0,0,0]
    angle = [0,0,0]
    angle[0] = angle[1] = 90
    angle[2] = 120
    length[0], length[2] = 5.97, 11.7
    length[1] = length[0]
    cell = CrystalCell()
    setCrystalCell(length, angle, cell)
    spaceGroup = SpaceGroup()
    setSpaceGroup("185", spaceGroup)    
    wavelength = 2.4437
    sMin, sMax = 0, 0.314
    return (spaceGroup, cell, wavelength, sMin, sMax)

def fit():
    wavelength = 2.4437
    spaceGroupName = "185"
    backg = LinSpline("LuFeO3_200K Background.BGR")
    tt, observed, unused = np.loadtxt("lufeo3_200k_2108.dat", dtype=float,
                                   unpack=True)
    cell = HexagonalCell(5.97, 11.7)
    cell.a.range(5,6)
    cell.c.range(11,12)
    #import sys; print >>sys.stderr, cell.c.bounds.limits
    m = Model(tt, observed, backg, 0, 0, 1, 1000, wavelength, spaceGroupName, cell)
    m.u.range(0,1)
    m.v.range(-1,0)
    m.w.range(0,10)
    [Ik.range(0,10000) for Ik in m.I]
    from bumps.names import FitProblem
    return FitProblem(m)

def main():
#    (spaceGroup, cell, wavelength, sMin, sMax) = inputInfo()
    (spaceGroup, cell, wavelength, sMin, sMax) = testInfo()
    reflections = hklGen(spaceGroup, cell, wavelength, sMin, sMax)
    printReflections(reflections, spaceGroup, wavelength, sMin, sMax)
    g = makeGaussians(reflections,[0, 0, 1], [1000]*len(reflections), wavelength)
    backg = LinSpline("LuFeO3_200K Background.BGR")
#    exclusions = np.array([[60,66],[72.75,75.75]])
    exclusions = None
    tt, observed, unused = np.loadtxt("lufeo3_200k_2108.dat", dtype=float,
                                   unpack=True)
    tt, observed = removeRange(tt, exclusions, observed)
    plotPattern(g, backg, tt, observed, twoTheta(sMin, wavelength),
                twoTheta(sMax, wavelength), .01, exclusions)
    pylab.show()
    return
    
#print >>sys.stderr,"cell created"
#spaceGroup = SpaceGroup()
#setSpaceGroup("185", spaceGroup)    
#print >>sys.stderr,"cell created"

if __name__ == "__main__":
    main()
else:
    from bumps.names import Parameter
    problem = fit()    

'''
Input data (LuFeO3):
185
5.97,11.7
2.4437
0,.314
'''
