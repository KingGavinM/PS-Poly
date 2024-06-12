#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3		// Use modern global access method and strict wave access.
//***************************************//
// PS-Poly                               //
// King Labratory                        //
// University of Missouri-Columbia 2024  //
// Written by: Liz Conley  	            //
// Email: elizabethconley@wustl.edu	     //
//***************************************//

// Contents:
// I. 		Main Functions
// II. 	Preprocessing
// III. 	Particle Measurements & Persistence Length Calculation
// IV.		Utilities
// V. 		Noise Cleaning Functions


//********************************************************************************************************//
//*********************** 			I. Main Functions				****************************************//
//*******************************************************************************************************//

//Main PS-Poly function that uses a single threshold to make the mask
//Make sure to select the data folder containing the images in the data browser
Function/S PSPOLY([ManualThreshold,pixscalefact])
 Variable ManualThreshold,pixscalefact
 String ImagesDFStr = GetBrowserSelection(0)
    String CurrentDFStr = GetDataFolder(1)
    If (!DataFolderExists(ImagesDFStr) || CountObjects(ImagesDFStr, 1) < 1) 
        DoAlert 0, "Select the folder with your images in it in the data browser, then try again."
        Return ""
    EndIf
    DFREF ImagesDF = $ImagesDFStr
    DFREF CurrentDF = $CurrentDFStr
    variable numImages = CountObjectsDFR(ImagesDF, 1)
 			
    setdatafolder root:
    newdatafolder/o/s root:Output
    DFREF output=root:output
		make/o/n=1 Xdata
		make/o/n=1 Ydata
		make/o/n=(numImages) TotalLengths
		
		
make/o/n=(1,2) All_branches
make/o/n=(1,2) All_Endpoints
variable ntotal=0
if(paramisdefault(ManualThreshold)==0)
	make/o/n=(numImages) thresholds
	variable t
	for(t=0;t<dimsize(thresholds,0);t++)
		thresholds[t]=ManualThreshold
	endfor
else
wave thresholds=GetThresholds(imagesDF)
endif
if(paramisdefault(pixscalefact)!=0)
	pixscalefact=4
endif

variable w,f=0,g=0,index=0,m=0,v
    for(w=0;w<numImages;w++)
    variable ind=w+1
    print "*** PROCESSING IMAGE ",ind," of ",numImages," ***"
    		killwaves/z mask,skeleton
			wave image = WaveRefIndexedDFR(ImagesDF, w)
			string imname=nameofwave(image)
	
			newdatafolder/o/s $imName
			DFREF imDF=$GetDatafolder(1)
			variable thresh=thresholds[w]
			wave mask=maskThreshold(image,thresh)
			wave bigmask=SizeUp(mask,pixScaleFact)
			string maskname=imname+"_mask"
			duplicate/o bigmask,$maskname
			
			make/o/n=1 imxdata
			make/o/n=1 imydata
			wave skeletonRaw=getlinesmask(mask)
			wave skeleton=CleanSkeletonEdges(skeletonRaw,5)
			string skelname=imname+"_skeleton"
			duplicate/o skeleton,$skelname
			
			wave HeightSkeleton=MakeHeightSkeleton(image,skeleton)
			
			wave imbranches=FindBranches(skeleton,HeightSkeleton)
			for(v=0;v<dimsize(imbranches,0);v++)
			insertpoints/m=0 f,1,all_branches
			all_branches[f][0]=imbranches[v][0]
			all_branches[f][1]=imbranches[v][1]
			f+=1
			endfor
			string imbranchname=imname+"_branches"
			duplicate/o imbranches,$imbranchname
			killwaves/z imbranches
			wave endpoints1=FindEndpoints(skeleton)
			string imEndName=imname+"_endpoints"
			for(v=0;v<dimsize(endpoints1,0);v++)
			insertpoints/m=0 g,1,all_endpoints
			all_endpoints[g][0]=endpoints1[v][0]
			all_endpoints[g][1]=endpoints1[v][1]
			g+=1
			endfor
			duplicate/o endpoints1, $imEndname
			killwaves/z endpoints1

clearsinglepixelnoise(skeleton)
wave seeds=SeedbyFill(skeleton,0)
variable nSeeds=Dimsize(seeds,0)
ntotal+=nSeeds
make/o/n=(nSeeds) ContourLengths
make/o/n=(nSeeds) numPix
make/o/n=(nSeeds) CountConfirm
make/o/n=(500,nseeds) xlocations
make/o/n=(500,nseeds)ylocations
ProcessAllSeeds(HeightSkeleton,countconfirm,ContourLengths,numpix,skeleton,seeds,xlocations,ylocations)

			newDataFolder/o Noise
			DFREF Noise=imDF:Noise
			newDataFolder/o ShortLinear
			DFREF ShortLinear=imDF:ShortLinear
			newDatafolder/o Linear
			DFREF Linear=imDF:Linear
			newDatafolder/o Branched
			DFREF Branches=imDF:Branched
			newDatafolder/o Loops
			DFREF Loops=imDF:Loops
			newDataFolder/o HighPoints
			DFREF HighPoints=imDF:HighPoints
			newDatafolder/o Complx
			DFREF Complx=imDF:Complx
			newDatafolder/o Killed
			DFREF Killed=imDF:Killed
			newDataFolder/o Overlapped
			DFREF Overlapped=imDF:Overlapped


variable n
variable nnoise=0,nPretzel=0,totalPretzel=0,nHP=0,totalHP=0,nlinear=0,totallinear=0,nloop=0,totalloop=0,nbranch=0,totalbranch=0,ncomplx=0,totalcomplx=0,nOverlap=0,totalOverlap=0
variable imtotal=0

variable q=0
for(n=0;n<nSeeds;n++)
duplicate/o image,im
	variable x=seeds[n][0]
	variable y=seeds[n][1]
	wave locations=ExtractLocations(n,xlocations,ylocations)
	wave limits=FindLimits(skeleton,locations)
	string locname=Uniquename("Locations_",1,n)
	make/o/n=1 lentemp
	lentemp[0]=ContourLengths[n]
	variable check=CountConfirm[n]
	wave Endpts=LookForEndpoints(skeleton,locations)
	wave branc=LookForBranches(skeleton,HeightSkeleton,locations)
	wave knots=LookForKnots(HeightSkeleton,locations)
	variable type=TestParticleType(skeleton,HeightSkeleton,branc,endpts,locations)
	string pskelname=Uniquename("skeleton_",1,n)
	string Pname=Uniquename("particle_",11,n)
	string croppname=Uniquename("particle_",1,n)
	string lenName=Uniquename("Particle_Length_",1,n)
	string endname=Uniquename("Endpoints_",1,n)
	string branchname=Uniquename("Branches_",1,n)
	string knotname=Uniquename("Overlap_point_",1,n)
	
	if(HasHighPoints(HeightSkeleton,locations)==1)
				setdatafolder HighPoints
				newdatafolder/o/s $pname
				nHP+=1
				totalHp+=lentemp[0]
				
				duplicate/o lenTemp,$lenName
				duplicate/o locations,$locname
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname
		setdatafolder output
	endif	
		
		if(type==0)
				setdatafolder noise
				nnoise+=1
				newdatafolder/o/s $pname
				duplicate/o lenTemp,$lenName
				duplicate/o locations,$locname
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname
		elseif(type==3 && check==1)//linear
				nlinear+=1
				totallinear+=lentemp[0]
				imtotal+=lentemp[0]
			//persistence length variables
				insertpoints/m=0 m,1,xdata
				insertpoints/m=0 q,1,imxdata
				xdata[m]=lentemp[0]
				imxdata[q]=lentemp[0]
				variable x1=endpts[0][0]
				variable y1=endpts[0][1]
				variable x2=endpts[1][0]
				variable y2=endpts[1][1]
				variable xETE=(x2-x1)*dimdelta(image,0)
				variable yETE=(y2-y1)*dimdelta(image,1)
				variable ETE=sqrt((xETE^2)+(yETE^2))^2
				insertpoints/m=0 m,1,ydata
				insertpoints/m=0 q,1,imydata
				ydata[m]=ETE
				imydata[q]=ETE
				m++
				q++
	
			//store particle info
				setdatafolder linear
				newdatafolder/o/s $pname //data foldr for particle
				duplicate/o endpts,$endname
				duplicate/o lenTemp,$lenName
				duplicate/o locations,$locname
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname
							
										
		elseif(type==4 && check==1) //loop
				SetDataFolder Loops
				nloop+=1
				totalloop+=lentemp[0]
				imtotal+=lentemp[0]
				newdatafolder/o/s $pname
				duplicate/o lenTemp,$lenName
				duplicate/o locations,$locname
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname

		elseif(type==5 && check==1)
			 //linear branching
				SetDataFolder Branches
				nbranch+=1
				imtotal+=lentemp[0]
				totalbranch+=lentemp[0]
				newdatafolder/o/s $pname
				duplicate/o endpts,$endname
				duplicate/o Branc,$branchname
				duplicate/o lenTemp,$lenName
				duplicate/o locations, $locname								
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname
	
		elseif(type==6 && check==1)
				setdatafolder Complx
				nPretzel+=1
				totalPretzel+=lentemp[0]
				imtotal+=lentemp[0]
				newdatafolder/o/s $pname
				duplicate/o endpts,$endname
				duplicate/o branc,$branchname
				duplicate/o lentemp,$lenname
				duplicate/o locations,$locname
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partskel,$pskelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname

		elseif(type==2 && check==1)
				SetDataFolder ShortLinear
				imtotal+=lentemp[0]
				nlinear+=1
				totallinear+=lentemp[0]
				newdatafolder/o/s $pname
				duplicate/o endpts,$endname
				duplicate/o lenTemp,$lenName
				duplicate/o locations,$locname
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname
	
		elseif(type==1 && check==1)
				setdatafolder Overlapped
				nOverlap+=1
				totalOverlap+=lentemp[0]
				imtotal+=lentemp[0]
				newdatafolder/o/s $pname
				duplicate/o endpts,$endname
				duplicate/o locations,$locname							
				duplicate/o lenTemp,$lenName
				duplicate/o branc, $branchname
				duplicate/o knots,$knotname
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname
			
		elseif(check==0)
					SetDataFolder Killed	
					duplicate/o lenTemp,$lenName
					duplicate/o locations,$locname							
				wave partSkel=CropParticle(skeleton,limits,10)
				duplicate/o partSkel,$pSkelname
				wave particle=CropParticle(image,limits,10)
				duplicate/o particle,$croppname

							
		
		endif
	killwaves/z particle,locations,limits,lentemp,knots,endpoints,branches
	setdatafolder imdf
		
endfor
		killwaves/z linesSkeleton
		killdatafolder Killed
		trimpoints(imxdata)
		trimpoints(imydata)
		duplicate/o BigMask,ExpandedMask
		killwaves/z seeds,bigmask
		string plname=imname+"_p_length"
			variable im_side_length
		if(dimsize(image,0)>=dimsize(image,1))
		im_side_length=dimsize(image,0)*dimdelta(image,0)
		else
		im_side_length=dimsize(image,1)*dimdelta(image,1)
		endif
		variable P_len=findBestfitETEvCL(imxdata,imydata,im_side_length,.1e-9)
		make/o/n=1 ptemp
		ptemp[0]=p_len
		duplicate/o ptemp,$plname
		killwaves/z ptemp,ypts,tempwave,limits,costs,branc,knots
TotalLengths[w]=imtotal
string imlengthsname=imname+"_Lengths"
duplicate/o ContourLengths,$imlengthsname
killwaves/z BranchStart,CountConfirm,contourlengths,endpoints,im,lentemp,skeleton,locations,xlocations,ylocations,numpix,mask,branches
setdatafolder output

endfor
make/o/n=1 Persistence_Length_nm
make/o/n=1 PL_error_nm
killwaves/z thresholds
trimpoints(all_branches)
trimpoints(all_endpoints)
trimpoints(xdata)
trimpoints(ydata)

variable avgLinear=totalLinear/nlinear
variable plinear=(totallinear/imtotal)*100
variable avgLoop=totalLoop/nloop
variable pLoop=(totalLoop/imtotal)*100
variable avgBranch=(totalbranch/nbranch)
variable pBranch=(totalbranch/imtotal)*100
variable avgPretzel=totalPretzel/npretzel
variable pPretzel=(totalPretzel/imtotal)*100
variable avgOverlap=totalOverlap/noverlap
variable poverlap=(totalOverlap/imtotal)*100
variable totalPercent=pLinear+pBranch+pPretzel+pLoop+pOverlap
print "Average Linear: ",avgLinear
print "percent Linear: ",pLinear
print "Average Loop: ",avgLoop
print "percent Loop: ",pLoop
print "Average Branched: ",avgBranch
print "Percent Branched: ",pbranch
print "Average Pretzel: ",avgPretzel
print "Percent Pretzel: ",pPretzel
print "Average Overlap: ",avgOverlap
print "Percent Overlap: ",pOverlap
print "Total Percent: ",totalPercent
print "Number of High Points: ",nHP
print "Number of Noise particles: ",nnoise


print "***Calculating Persistence Length***"
	Display Ydata vs Xdata
	ModifyGraph mode=2,lsize=3
	

	

	if(dimsize(image,0)>=dimsize(image,1))
		im_side_length=dimsize(image,0)*dimdelta(image,0)
	else
		im_side_length=dimsize(image,1)*dimdelta(image,1)
	endif
	
	
	// "guess" input igor makes you do for curve fitting
	Make/D/N=1/O W_coef
	//	find P vlaue with best fit by searching a range of lengths from 0 to size of image
	//	with .5 nm step size
	variable P_length=findBestfitETEvCL(xdata,ydata,im_side_length,.5e-9)
	W_coef[0] = {(P_length)}
	
	
	//fit the curve and print values
	FuncFit fitCLvsETE W_coef Ydata /X=Xdata /D
	print "P= ",w_coef[0]
	FuncFit fitCLvsETE W_coef Ydata /X=Xdata /D
	wave  w_sigma1=:W_sigma
	variable pl=w_coef[0]*1e9
	make/o/n=1 Persistence_length_nm
	Persistence_length_nm[0]=pl
	make/o/n=1 PL_Error_nm
	PL_Error_nm[0]=w_sigma1[0]*1e9
	
	Label left "RMS End-to-End Distances"
	Label bottom "Contour Length (nm)"
	TextBox/C/N=text0/F=1/A=MC "Lp= "+num2str(pl)+" +/- "+num2str(PL_ERROR_nm[0])+" nm"
	print "persistence length= ",pl,"+/-",PL_Error_nm[0]," nm"
	print "Number of particles in PL calculation= ",nlinear

	killwaves/z costs,ypts,tempwave,w_coef,w_sigma
	SetDataFolder Output
	string OutputStr=GetDataFolder(1)
	return OutputStr


end

//*******************************************************************************************************//
//*********************** 			II. Preprocessing				****************************************//
//*******************************************************************************************************//

//prompts user with slider to select a threshold for each image
function/WAVE GetThresholds(df)
DFREF df
variable size=CountObjectsDFR(df,1)
make/o/n=(size) Thresholds
variable n
for(n=0;n<size;n++)
wave image = WaveRefIndexedDFR(df,n)
variable thresh=interactiveHeightThreshold(image)
thresholds[n]=thresh
endfor
return thresholds
end

//creates a floodmask with the specified height threshold 
function/wave MaskThreshold(im,threshold)
wave im
variable threshold
variable delta_x=dimdelta(im,0)
variable delta_y=dimdelta(im,1)
variable i,j
make/o/n=(dimsize(im,0),dimsize(im,1)) mask
for(i=0;i<dimsize(im,0);i++)
for(j=0;j<dimsize(im,1);j++)
	if(im[i][j]>=threshold)
		mask[i][j]=1
	else
		mask[i][j]=0
	endif
endfor
endfor
setscale/p x,0,delta_x,"m",mask
setscale/p y,0,delta_y,"m",mask
return mask
end

//performs skeletonization on floodmask
Function/WAVE GetLinesMask(mask)
wave mask
string name
duplicate/o mask, preskeleton
	variable p,q,n
	Redimension/B/U/N=(dimsize(PreSkeleton,0),dimsize(Preskeleton,1),3) PreSkeleton

		For (p=0;p<dimsize(Preskeleton,0);p++)
		For (q=0;q<dimsize(Preskeleton,1);q++)
			Preskeleton[p][q][1]= Preskeleton[p][q][0]
			Preskeleton[p][q][2]= Preskeleton[p][q][0]
		endfor
		endfor
	 
	Duplicate/o PreSkeleton, LinesSkeleton

	Imageskeleton3D/DEST=LinesSkeleton Preskeleton

		For(p=0; p<dimsize(LinesSkeleton,0);p++)
		For(q=0; q<dimsize(LinesSkeleton,1);q++)
			LinesSkeleton[p][q][0]=LinesSkeleton[p][q][1]
			LinesSkeleton[p][q][2]=LinesSkeleton[p][q][1]
		endfor
		endfor
	
	Redimension/N=(dimsize(LinesSkeleton,0),dimsize(LinesSkeleton,1)) LinesSkeleton

	Killwaves/Z Preskeleton
	
	return linesSkeleton
	
end



//makes a skeleton with the height information from the original image
Function/WAVE MakeHeightSkeleton(im,skeleton)
wave im,skeleton
make/o/n=(dimsize(im,0),dimsize(im,1)) HeightSkeleton
variable totalHeight=0
variable i,j
variable pixels=countpixelsim(skeleton)
for(i=0;i<dimsize(im,0);i++)
for(j=0;j<dimsize(im,1);j++)
	if(skeleton[i][j]==1)
		totalHeight+=(1e9)*im[i][j]
		HeightSkeleton[i][j]= (1e9)*im[i][j]
	endif
endfor
endfor
variable avg=totalHeight/pixels
note HeightSkeleton, "Average:"+num2str(avg)
return HeightSkeleton
end

Function InteractiveHeightThreshold(image)
	Wave image
	
	
	duplicate/O image, im
	
	
	Make/N=(DimSize(image,0),DimSize(image,1)) /O im
	SetScale/P x,DimOffset(image,0),DimDelta(image,0), im
	SetScale/P y,DimOffset(image,1),DimDelta(image,1), im
	
	Multithread im = image[p][q][0]
	duplicate/O im, preview
	
	//Set Ranges
	Variable MaxH= wavemax(im)
	Variable StartV= (MaxH)/2

	duplicate/O im H_MAXMAP
	wave map = H_MaxMap
	Multithread map= -1
	Duplicate/o map, H_MAXScaleMap
	wave scalemap = H_MAXScaleMAp
	Variable lowlim= wavemin(im), uplim=wavemax(im)
	variable dH = (uplim-lowlim)/100
	
	//display image & masking duplicate
	Newimage/N=IMAGE preview
	
	//Quit Variable
	Variable/G InteractiveQuit=0
	
	//Panel
	NewPanel/EXT=0 /HOST=IMAGE /N=Subcontrol /W= (0,0,100,550) as "Continue Button"
	Button btn pos={0,0}, size={100,50}, title="Accept", win=IMAGE#SubControl, proc=InteractiveContinue
	Button quitbtn pos={0,50}, size={100,50}, title="Quit", win=IMAGE#SubControl, proc=InteractiveQuit
	Variable/G H_Thresh=StartV
	Slider HThresh limits={lowlim,uplim,dH}, pos= {0,100},size={100,400}, variable=H_Thresh, proc=HeightSlider
	//ValDisplay threshold variable=H_thresh,pos={50,300}

	Pauseforuser IMAGE
	
	
	variable quitvar =Interactivequit
	
	
	variable returnval=H_thresh
	Killvariables/Z H_Thresh,Interactivequit
	KillWaves/Z Map, original,h_maxscalemap
	
	
	
	print "Chosen Height Threshold:", returnval
	killwaves/z im
	note preview, "Threshold:"+Num2Str(returnval)
	killwaves/z preview
	return returnval
	
end


//****************************************************************************************************//
//***************   III. 	Particle Measurements & Persistence Length Calculation   *******************//
//****************************************************************************************************//


//determines all branch points within an image and removes overlap points from count
function/WAVE FindBranches(skeleton,HeightSkeleton)
wave skeleton,HeightSkeleton
string name
variable i,j
make/o/n=(1,2) branches
variable n=0
	for(i=0;i<dimsize(skeleton,0);i++)
	for(j=0;j<dimsize(skeleton,1);j++)
		if(skeleton[i][j]==1)
			if(isBranch(skeleton,i,j)==1 && isKnot(HeightSkeleton,i,j)==0)
				insertpoints/m=0 n,1,branches
				branches[n][0]=i
				branches[n][1]=j
				n++
			endif
		endif
	endfor
	endfor
variable size=dimsize(branches,0)
deletepoints/m=0 (size-1),1,branches

return branches
end

//Determines if the test pixel is an overlap point
//im is HeightSkeleton
threadsafe Function isKnot(im,x,y)
wave im
variable x,y
	if(x==0 || y==0)
		return 0
	endif
	wave NeighborPath=IsKnotNeighborPAth(im,x,y)
	variable i
	variable path_count=0
	for(i=1;i<9;i++)
		if(im[x][y]==1)
			if(neighborPath[i]==1 && neighborpath[i-1]==0 &&neighborpath[i+1]==0)
				path_count+=1
			endif
		endif
	endfor
	killwaves/z neighborpath
	if(path_count>=3)
		string avgStr=GetNote(im,"Average")
		variable average=str2num(avgStr)
		if(im[x][y]>=(1.5*average))
			return 1
		else
			return 0
		endif
	else
		return 0
	endif
	
end

//if over 80% of the particle is over 1.5 times the average height of the rest of the image, it is noise
threadsafe Function isNoise(HeightSkeleton,Locations)
wave HeightSkeleton,locations
variable n
variable average=str2num(GetNote(HeightSkeleton,"Average"))
variable threshold=1.5*average
variable AboveCount=0
variable total=dimsize(locations,0)
for(n=0;n<dimsize(locations,0);n++)
variable x=locations[n][0]
variable y=locations[n][1]
	if(HeightSkeleton[x][y]>=Threshold)
		AboveCount+=1
	endif
endfor
variable percentAbove=AboveCount/total
if(percentAbove>=.8)
	return 1
else
	return 0
endif

end


//returns a list of all of the endpoints contained within an image
function/WAVE FindEndpoints(skeleton)
wave skeleton
variable i,j
make/o/n=(1,2) Endpoints
variable n=0
	for(i=0;i<dimsize(skeleton,0);i++)
	for(j=0;j<dimsize(skeleton,1);j++)
		if(skeleton[i][j]==1)
			if(isEndpoint(skeleton,i,j)==1)
				insertpoints/m=0 n,1,endpoints
				endpoints[n][0]=i
				endpoints[n][1]=j
				n++
			endif
		endif
	endfor
	endfor
trimpoints(endpoints)

return endpoints
end

//determines if test point is an endpoint
threadsafe Function isEndpoint(skeleton,x,y)
	wave skeleton
	variable x,y
	
	DFREF currentDF= $GetDatafolder(1)
	NewDatafolder/S/o root:pixmaps
	make/o/n=(3,3) Endtype1
							endtype1[0][0]=0
							endtype1[0][1]=0
							endtype1[0][2]=0
							endtype1[1][0]=1
							endtype1[1][1]=1
							endtype1[1][2]=0
							endtype1[2][0]=0
							endtype1[2][1]=0
							endtype1[2][2]=0
					
	make/o/n=(3,3) Endtype2
							endtype2[0][0]=0
							endtype2[0][1]=0
							endtype2[0][2]=0
							endtype2[1][0]=0
							endtype2[1][1]=1
							endtype2[1][2]=0
							endtype2[2][0]=0
							endtype2[2][1]=1
							endtype2[2][2]=0
	make/o/n=(3,3) Endtype3
							endtype3[0][0]=0
							endtype3[0][1]=0
							endtype3[0][2]=0
							endtype3[1][0]=0
							endtype3[1][1]=1
							endtype3[1][2]=1
							endtype3[2][0]=0
							endtype3[2][1]=0
							endtype3[2][2]=0
	make/o/n=(3,3) Endtype4
							endtype4[0][0]=0
							endtype4[0][1]=1
							endtype4[0][2]=0
							endtype4[1][0]=0
							endtype4[1][1]=1
							endtype4[1][2]=0
							endtype4[2][0]=0
							endtype4[2][1]=0
							endtype4[2][2]=0
	make/o/n=(3,3) Endtype5
							endtype5[0][0]=0
							endtype5[0][1]=0
							endtype5[0][2]=0
							endtype5[1][0]=0
							endtype5[1][1]=1
							endtype5[1][2]=0
							endtype5[2][0]=1
							endtype5[2][1]=0
							endtype5[2][2]=0
	make/o/n=(3,3) Endtype6
							endtype6[0][0]=1
							endtype6[0][1]=0
							endtype6[0][2]=0
							endtype6[1][0]=0
							endtype6[1][1]=1
							endtype6[1][2]=0
							endtype6[2][0]=0
							endtype6[2][1]=0
							endtype6[2][2]=0
	make/o/n=(3,3) Endtype7
							endtype7[0][0]=0
							endtype7[0][1]=0
							endtype7[0][2]=1
							endtype7[1][0]=0
							endtype7[1][1]=1
							endtype7[1][2]=0
							endtype7[2][0]=0
							endtype7[2][1]=0
							endtype7[2][2]=0
	make/o/n=(3,3) Endtype8
							endtype8[0][0]=0
							endtype8[0][1]=0
							endtype8[0][2]=0
							endtype8[1][0]=0
							endtype8[1][1]=1
							endtype8[1][2]=0
							endtype8[2][0]=0
							endtype8[2][1]=0
							endtype8[2][2]=1
	make/o/n=(3,3) Endtype9
							endtype9[0][0]=0
							endtype9[0][1]=0
							endtype9[0][2]=1
							endtype9[1][0]=0
							endtype9[1][1]=1
							endtype9[1][2]=1
							endtype9[2][0]=0
							endtype9[2][1]=0
							endtype9[2][2]=0
	make/o/n=(3,3) Endtype10
							endtype10[0][0]=0
							endtype10[0][1]=0
							endtype10[0][2]=0
							endtype10[1][0]=0
							endtype10[1][1]=1
							endtype10[1][2]=1
							endtype10[2][0]=0
							endtype10[2][1]=0
							endtype10[2][2]=1
	make/o/n=(3,3) Endtype11
							endtype11[0][0]=0
							endtype11[0][1]=1
							endtype11[0][2]=1
							endtype11[1][0]=0
							endtype11[1][1]=1
							endtype11[1][2]=0
							endtype11[2][0]=0
							endtype11[2][1]=0
							endtype11[2][2]=0
	make/o/n=(3,3) Endtype12
							endtype12[0][0]=1
							endtype12[0][1]=1
							endtype12[0][2]=0
							endtype12[1][0]=0
							endtype12[1][1]=1
							endtype12[1][2]=0
							endtype12[2][0]=0
							endtype12[2][1]=0
							endtype12[2][2]=0
	make/o/n=(3,3) Endtype13
							endtype13[0][0]=1
							endtype13[0][1]=0
							endtype13[0][2]=0
							endtype13[1][0]=1
							endtype13[1][1]=1
							endtype13[1][2]=0
							endtype13[2][0]=0
							endtype13[2][1]=0
							endtype13[2][2]=0
	make/o/n=(3,3) Endtype14
							endtype14[0][0]=0
							endtype14[0][1]=0
							endtype14[0][2]=0
							endtype14[1][0]=1
							endtype14[1][1]=1
							endtype14[1][2]=0
							endtype14[2][0]=1
							endtype14[2][1]=0
							endtype14[2][2]=0
	make/o/n=(3,3) Endtype15
							endtype15[0][0]=0
							endtype15[0][1]=0
							endtype15[0][2]=0
							endtype15[1][0]=0
							endtype15[1][1]=1
							endtype15[1][2]=0
							endtype15[2][0]=1
							endtype15[2][1]=1
							endtype15[2][2]=0
	make/o/n=(3,3) Endtype16
							endtype16[0][0]=0
							endtype16[0][1]=0
							endtype16[0][2]=0
							endtype16[1][0]=0
							endtype16[1][1]=1
							endtype16[1][2]=0
							endtype16[2][0]=0
							endtype16[2][1]=1
							endtype16[2][2]=1
							
	
	variable i,j
	
	setdatafolder currentdf
	setdatafolder root:pixmaps
	variable n=0
	variable result=0
			wave pixmap= GeneratePixMap(Skeleton,x,y)
			variable p,q
			variable end1=0, end2=0, end3=0, end4=0,end5=0,end6=0,end7=0,end8=0,end9=0,end10=0,end11=0,end12=0,end13=0,end14=0,end15=0,end16=0
	
				For(p=0;p<3;p++)
				For(q=0;q<3;q++)
					If(pixmap[p][q]==endtype1[p][q])
						end1+=1
						endif
					If(pixmap[p][q]==endtype2[p][q])
						end2+=1
						endif
					If(pixmap[p][q]==endtype3[p][q])
						end3+=1
						endif
					If(pixmap[p][q]==endtype4[p][q])
						end4+=1
						endif
					If(pixmap[p][q]==endtype5[p][q])
						end5+=1
						endif
					If(pixmap[p][q]==endtype6[p][q])
						end6+=1
						endif
					If(pixmap[p][q]==endtype7[p][q])
						end7+=1
						endif
					If(pixmap[p][q]==endtype8[p][q])
						end8+=1
						endif
					If(pixmap[p][q]==endtype9[p][q])
						end9+=1
						endif
					If(pixmap[p][q]==endtype10[p][q])
						end10+=1
						endif
					If(pixmap[p][q]==endtype11[p][q])
						end11+=1
						endif
					If(pixmap[p][q]==endtype12[p][q])
						end12+=1
						endif
					If(pixmap[p][q]==endtype13[p][q])
						end13+=1
						endif
					If(pixmap[p][q]==endtype14[p][q])
						end14+=1
						endif
					If(pixmap[p][q]==endtype15[p][q])
						end15+=1
						endif
					If(pixmap[p][q]==endtype16[p][q])
						end16+=1
					endif
				Endfor
				endfor
				If(end1==9||end2==9||end3==9||end4==9||end5==9||end6==9||end7==9||end8==9||end9==9||end10==9||end11==9||end12==9||end13==9||end14==9||end15==9||end16==9)
					result=1	
				endif			
				

	setdatafolder currentDF
	killdatafolder/Z root:pixmaps
	killwaves/z pixmap
	
	return result
	
	
end


//returns a list of all of the points of overlap within an image
function/WAVE FindKnots(Heightskeleton)
wave Heightskeleton
variable i,j
make/o/n=(1,2) Knots
variable n=0
	for(i=0;i<dimsize(Heightskeleton,0);i++)
	for(j=0;j<dimsize(Heightskeleton,1);j++)
		if(Heightskeleton[i][j]!=0)
			if(isKnot(Heightskeleton,i,j)==1)
				insertpoints/m=0 n,1,Knots
				knots[n][0]=i
				knots[n][1]=j
				n++
			endif
		endif
	endfor
	endfor
trimpoints(knots)

return knots
end

// extracts individual particle coordinates from output of seed processing
Function/WAVE ExtractLocations(SeedIndex,xlocations,ylocations)
variable SeedIndex
wave xlocations
wave ylocations
make/o/n=(1,2) locations
variable n,r=0
for(n=0;n<dimsize(xlocations,0);n++)
	variable x=xlocations[n][SeedIndex]
	variable y=ylocations[n][SeedIndex]
	if( x!=0 && y!=0)
		insertpoints/m=0 r,1,locations
		locations[r][0]=x
		locations[r][1]=y
		r++
	endif
endfor
trimpoints(locations)
return locations

end

Function/WAVE FindLimits(im,locations)
wave im,locations
variable i,j

variable xlow=dimsize(im,0),ylow=dimsize(im,1),xhigh=0,yhigh=0

	variable n
	for(n=0;n<dimsize(locations,0);n++)
	i=locations[n][0]
	j=locations[n][1]
	
			if( i < xlow)
				xlow=i
			endif
			if(i>xhigh)
				xhigh=i
			endif
			if(j < ylow)
				ylow=j
			endif
			if(j > yhigh)
				yhigh=j
			endif
		
	endfor
	
make/o/n=(2,2) Limits
limits[0][0]=xlow
limits[0][1]=ylow
limits[1][0]=xhigh
limits[1][1]=yhigh
killwaves/z visited
return limits
end

//finds endpoints listed within locations wave
threadsafe function/WAVE LookForEndpoints(skeleton,locations)
wave skeleton,locations
make/o/n=(1,2) Endpoints
variable n=0
variable r
for(r=0;r<dimsize(locations,0);r++)
	variable x=locations[r][0]
	variable y=locations[r][1]
	if(isEndpoint(skeleton,x,y)==1)
		insertpoints/m=0 n,1,endpoints
		endpoints[n][0]=x
		endpoints[n][1]=y
		n++
	endif
endfor
variable size=dimsize(endpoints,0)
deletepoints/m=0 (size-1),1,endpoints
return endpoints
end

//finds branches listed in locations wave
Threadsafe Function/WAVE LookForBranches(im,HeightSkeleton,locations)
wave im,locations,HeightSkeleton
variable r
make/o/n=(1,2) Branches
variable n=0
for(r=0;r<dimsize(locations,0);r++)
	variable x=Locations[r][0]
	variable y=Locations[r][1]
	if(isBranch(im,x,y)==1 && isKnot(HeightSkeleton,x,y)==0)
			insertpoints/m=0 n,1,branches
			Branches[n][0]=x
			Branches[n][1]=y
			n++
	endif
endfor
variable size=dimsize(branches,0)
deletepoints/m=0 (size-1),1,branches
return branches
end

//finds overlap points listed in locations wave
threadsafe function/WAVE LookForKnots(Heightskeleton,locations)
wave Heightskeleton,locations
make/o/n=(1,2) Knots
variable n=0
variable r
for(r=0;r<dimsize(locations,0);r++)
	variable x=locations[r][0]
	variable y=locations[r][1]
	if(isKnot(Heightskeleton,x,y)==1)
		insertpoints/m=0 n,1,knots
		knots[n][0]=x
		knots[n][1]=y
		n++
	endif
endfor
variable size=dimsize(knots,0)
deletepoints/m=0 (size-1),1,knots
return knots
end

//finds path around test point to test for an overlap
threadsafe Function/WAVE IsKnotNeighborPath(im,xseed,yseed)
wave im
variable xseed,yseed
make/o/n=10 NeighborPath

if( (xseed+1)>dimsize(im,0) || (yseed+1)>dimsize(im,0) || (xseed-1)<0 || (yseed-1)<0)
	return neighborPath
endif
	if(im[xSeed][yseed+1]!=0)
		neighborpath[0]=1
	else
		neighborpath[0]=0
	endif
	
	if(im[xSeed+1][yseed+1]!=0)
		neighborpath[1]=1
	else 
		neighborpath[1]=0
	endif
	
	if(im[xseed+1][yseed]!=0)
		neighborpath[2]=1
	else
		neighborpath[2]=0
	endif
	
	if(im[xseed+1][yseed-1]!=0)
		neighborpath[3]=1
	else
		neighborpath[3]=0
	endif
	
	if(im[xseed][yseed-1]!=0)
		neighborpath[4]=1
	else
		neighborpath[4]=0
	endif
	
	if(im[xseed-1][yseed-1]!=0)
		neighborpath[5]=1
	else
		neighborpath[5]=0
	endif
	
	if(im[xseed-1][yseed]!=0)
		neighborpath[6]=1
	else
		neighborpath[6]=0
	endif
	
	if(im[xSeed-1][yseed+1]!=0)
		neighborpath[7]=1
	else
		neighborpath[7]=0
	endif
	
	if(im[xseed][ySeed+1]!=0)
		neighborpath[8]=1
	else
		neighborpath[8]=0
	endif
	
	if(im[xseed+1][yseed+1]!=0)
		neighborpath[9]=1
	else
		neighborpath[9]=0
	endif
	
	return neighborpath
end


//determines if test point is a branch
threadsafe Function isBranch(im,x,y)
wave im
variable x,y
	if(x==0 || y==0)
		return 0
	endif
	wave NeighborPath=IsBranchNeighborPAth(im,x,y)
	variable i
	variable path_count=0
	for(i=1;i<9;i++)
		if(im[x][y]==1)
			if(neighborPath[i]==1 && neighborpath[i-1]==0 &&neighborpath[i+1]==0)
				path_count+=1
			endif
		endif
	endfor
	for(i=1;i<8;i++)
		if(im[x][y]==1)
			if(neighborpath[i]==1 && neighborpath[i+1]==1 && neighborpath [i-1]==0 && neighborpath[i+2]==0)
				path_count+=1
			endif
		endif
	endfor
	killwaves/z neighborpath
	if(path_count>=3)
		return 1
	else
		return 0
	endif
	
end

//finds clockwise path around test point to test for branch points
threadsafe Function/WAVE IsBranchNeighborPath(im,xseed,yseed)
wave im
variable xseed,yseed
make/o/n=10 NeighborPath

if( (xseed+1)>dimsize(im,0) || (yseed+1)>dimsize(im,0) || (xseed-1)<0 || (yseed-1)<0)
	return neighborPath
endif
	if(im[xSeed][yseed+1]==1)
		neighborpath[0]=1
	else
		neighborpath[0]=0
	endif
	
	if(im[xSeed+1][yseed+1]==1)
		neighborpath[1]=1
	else 
		neighborpath[1]=0
	endif
	
	if(im[xseed+1][yseed]==1)
		neighborpath[2]=1
	else
		neighborpath[2]=0
	endif
	
	if(im[xseed+1][yseed-1]==1)
		neighborpath[3]=1
	else
		neighborpath[3]=0
	endif
	
	if(im[xseed][yseed-1]==1)
		neighborpath[4]=1
	else
		neighborpath[4]=0
	endif
	
	if(im[xseed-1][yseed-1]==1)
		neighborpath[5]=1
	else
		neighborpath[5]=0
	endif
	
	if(im[xseed-1][yseed]==1)
		neighborpath[6]=1
	else
		neighborpath[6]=0
	endif
	
	if(im[xSeed-1][yseed+1]==1)
		neighborpath[7]=1
	else
		neighborpath[7]=0
	endif
	
	if(im[xseed][ySeed+1]==1)
		neighborpath[8]=1
	else
		neighborpath[8]=0
	endif
	
	if(im[xseed+1][yseed+1]==1)
		neighborpath[9]=1
	else
		neighborpath[9]=0
	endif
	
	return neighborpath
end


function findBestfitETEvCL(xdata,ydata,im_max_length,stepsize)
wave xdata,ydata
variable stepsize, im_max_length

//find range for test values
	
	//(only for P<CL)
	//variable maxval=wavemax(xdata)
		variable maxval=im_max_length



	variable i
	variable j=0

	//find number of test values
	variable numsteps=maxval/stepsize
	make/o/n=1 costs
	variable step

	//begin iteration of test values
	for(i=0;i<maxval;i+=stepsize)
	step=i*stepsize
	wave yfits=ETEvCLfitline(xdata,i) 
	insertpoints/m=0 j,1,costs//yet y fit line for each value
	costs[j]=CalculateErrorCLvEte(yfits,xdata,ydata,i) //calculate error for each test value
	
	//store corresponding persistence length in wavenote
	string Istr=num2str(i)
	string coststr=num2str(costs[j])
	note costs coststr+":"+istr


	j+=1
	endfor
	variable size=numpnts(costs)
	deletepoints/m=0 (size-1),1,costs
	//find minimum error
	variable lowestcost=wavemin(costs)

	//retrieve corresponding persistence length from minimum error value
	string lowcostr=num2str(lowestcost)
	string lowcost=getnote(costs,lowcostr)
	variable Pval=str2num(lowcost)

	print "Persistence legnth= ",Pval
	return Pval
end

//calculates Y values of a fit line using mean-square end to end distance versus contour length
//given the contour length and a given persistence length
function/WAVE ETEvCLfitline(xdata,P)
	wave xdata
	variable p

	variable j,count

	// make Y points wave
	count=dimsize(xdata,0)
	variable i
	make/o/n=(count) Ypts
	
	//calculate predicted y value
	for(i=0;i<count;i++)
	//j=4*p*xdata[i]*(1-((2*p)/xdata[i])*(1-exp((-xdata[i])/(2*p))))
	j=(2)*2*p*xdata[i]*(1-(((2)*p)/xdata[i])*(1-exp((-xdata[i])/((2)*p))))
	ypts[i]=j
	endfor
	
	return ypts
end

//returns error between actual and predicted y values using mean squared error
function CalculateErrorCLvEte(yfit,xdata,ydata,p)
	wave xdata,ydata,yfit
	variable p

	variable i
	variable count=dimsize(xdata,0)

	//store squared differences in a temporary wave
	make/o/n=(count) tempwave
	for(i=0;i<count;i++)
	tempwave[i]=(yfit[i]-ydata[i])^2
	endfor

	//add difference values together
	variable summ=0
	for(i=0;i<count;i++)
	summ+=tempwave[i]
	endfor
	variable cost

	// calculate error
	cost=(1/(2*count))*summ
	return cost
	end


//fit function for mean-square end-to-end distance used for creating a graph of the fit
//This is the equation that relates end-to-end distace, contour length, and persistence length
function fitCLvsETE(w,x): fitfunc 
wave w
variable x
// w[0]=p

return (2)*2*x*w[0]*(1-((2*w[0])/x)*(1-exp((-x)/(2*w[0]))))

end

Function/WAVE CropParticle(im,limits,layers)
wave im,limits
variable layers
variable big=dimsize(im,0)*2
variable xlow= limits[0][0]-layers
if(xlow<=0)
	xlow=0
endif
variable ylow= limits[0][1]-layers
if(ylow<=0)
	ylow=0
endif
variable xhigh= limits[1][0] + layers
if(xhigh>=(dimsize(im,0)-1))
	xhigh=dimsize(im,0)-1
endif
variable yhigh= limits[1][1] + layers
if(yhigh>=(dimsize(im,1)-1))
	yhigh=dimsize(im,1)-1
endif
duplicate/o im,particle
deletepoints/m=0 xhigh,big,particle
deletepoints/m=0 0,xlow,particle
deletepoints/m=1 yhigh,big,particle
deletepoints/m=1 0,ylow,particle

return particle


end

//Function used to sort particles by type
//0:noise
//1:knots
//2:SL
//3:Linear
//4:Loop
//5:Branched
//6:Pretzel
threadsafe Function TestParticleType(skeleton,HeightSkeleton,branches,endpoints,locations)
	wave Skeleton,branches,endpoints,HeightSkeleton,locations
	variable type
	variable x1,x2,y1,y2
	variable n
	variable numpix=dimsize(locations,0)
	if(isNoise(HeightSkeleton,locations))
		type=0
	elseif(CheckForKnots(HeightSkeleton,locations)==1)
		type=1
	elseif(dimsize(branches,0)==0 && dimsize(endpoints,0)==2)
		if(numPix<=5)
		type=2
		else
		type=3
		endif
	//if theres no endpoints and no branch points then its a loop
	elseif(dimsize(branches,0)==0 && dimsize(endpoints,0)==0)
		type=4
	else
		variable PB=PretzelOrBranched(endpoints,branches)
		if(PB==0)
		type=5
		else
		type=6
		endif
	endif
		
return type
end

Function HasHighPoints(HeightSkeleton,locations)
wave HeightSkeleton,locations

variable i,j,total=0,count=0,HP=0
for(i=0;i<dimsize(HeightSkeleton,0);i++)
for(j=0;j<dimsize(HeightSkeleton,1);j++)
	if(HeightSkeleton[i][j]!=0)
		total+=HeightSkeleton[i][j]
		count+=1
	endif
endfor
endfor
variable average=total/count
variable threshold=1.5*average
variable n
for(n=0;n<dimsize(locations,0);n++)
	variable x=locations[n][0]
	variable y=locations[n][1]
	if(HeightSkeleton[x][y]>threshold)
		HP=1
	endif
endfor

return HP

end

//finds overlap points listed in locations wave
threadsafe Function CheckForKnots(HeightSkeleton,locations)
wave HeightSkeleton,locations
variable n
variable check=0
for(n=0;n<dimsize(locations,0);n++)
variable x=locations[n][0]
variable y=locations[n][1]
	if(isKnot(HeightSkeleton,x,y)==1)
		check=1
	endif
endfor
return check
end

//Main particle tracing function
threadsafe Function/WAVE TraceParticle(HeightSkeleton,skeleton,locations)
wave skeleton,locations,HeightSkeleton
variable xseed=locations[0][0],yseed=locations[0][1]
variable numpix=dimsize(locations,0)
wave Endpoints=LookForEndpoints(skeleton,locations)
wave branches=LookForBranches(skeleton,HeightSkeleton,locations)
variable type=TestParticleType(skeleton,HeightSkeleton,branches,endpoints,locations)
if(type==2 || type==3 )
	wave TraceResult=TraceLinear(skeleton,endpoints,locations)
elseif(type==4)
	wave TraceResult=TraceCircles(skeleton,locations)
else
	wave TraceResult=TraceComplex(skeleton,locations)
endif
note traceresult,"type:"+num2str(type)
return TraceResult

end

threadsafe Function/WAVE TraceLinear(im,endpoints,locations)
wave im,endpoints
wave locations
variable x=endpoints[0][0]
variable y=endpoints[0][1]
//START EDITS HERE
Make/O/N=(DimSize(im, 0), DimSize(im, 1)) path
path = 0
path[x][y] = 1
variable n = 0
variable n_sides = 0
variable n_diag = 0
variable next_i = x
variable next_j = y
variable count = 0
variable ns_x = 0
variable ns_y = 0
variable nd = 0
variable found = 0
variable dx, dy
variable endtype=CheckEndpointType(im,x,y)
if(endtype==2||endtype==4||endtype==11||endtype==12||endtype==15||endtype==16)
ns_x+=1
elseif(endtype==1||endtype==3||endtype==9||endtype==10||endtype==13||endtype==14)
ns_y+=1
elseif(endtype==5||endtype==6||endtype==7||endtype==8)
nd+=1
endif

Do
    found = 0
    if (isBranch(im,x,y)==1)
        //Branch!
        break
    endif

    // Loop through all neighbors, checking that the pixels are uncounted to avoid double-counting. Loop through diagonals last.
    for (dx = -1; dx <= 1; dx += 1)
        for (dy = -1; dy <= 1; dy += 1)
            if (dx == 0 && dy == 0)
                continue
            endif

            // Check for out-of-bounds indices
            if (x + dx < 0 || x + dx >= DimSize(im, 0) || y + dy < 0 || y + dy >= DimSize(im, 1))
                continue
            endif

            if (im[x + dx][y + dy] == 1 && path[x + dx][y + dy] == 0)
                path[x + dx][y + dy] = 1
                n += 1

                if (abs(dx) + abs(dy) == 1) // side neighbors
                    if (dx != 0) // horizontal neighbor
                        ns_x += 1
                    else // vertical neighbor
                        ns_y += 1
                    endif
                else // diagonal neighbors
                    n_diag += 1
                endif

                next_i = x + dx
                next_j = y + dy
                count += 1
                found = 1
                break
            endif
        endfor
        if (found)
            break
        endif
    endfor

    if (!found) // No more neighbors found
        break
    endif

    // Re-initialize to the next pixel
    x = next_i
    y = next_j

    nd += n_diag
    // Re-initialize counting parameters
    n = 0
    n_sides = 0
    n_diag = 0
    


While (1)
 
killwaves/z path, count
variable delta_x=dimdelta(im,0)
variable delta_y=dimdelta(im,1)
variable CL_x=ns_x*delta_x
variable CL_y=ns_y*delta_y
variable CL_diag=nd*sqrt(delta_x^2+delta_y^2)
variable Contour_length = CL_x + CL_y + CL_diag
variable numpix=ns_x+ns_y+nd
make/o/n=6 TraceResult
TraceResult[0]=Contour_length
TraceResult[1]=numpix
TraceResult[2]=ns_x
TraceResult[3]=ns_y
TraceResult[4]=nd
TraceResult[5]=0
variable CheckTrace=CheckTraceResult(TraceResult,locations)
if(checkTrace==1)
	TraceResult[5]=1
else
	TraceResult[5]=0
endif
return TraceResult

end

//TraceResult Key:
//TraceResult[0]= Contour Length
//TraceResult[1]=numpix
//TraceResult[2]=xtotal
//TraceResult[3]=ytotal
//TraceResult[4]=dtotal
//TraceResult[5]=Length Verified (1 if yes, 0 if no)
threadsafe Function/WAVE TraceCircles(skeleton,locations)
wave skeleton,locations
variable x=locations[0][0]
variable y=locations[0][1]
variable StartX=x
variable StartY=y
variable n = 0
variable n_sides = 0
variable n_diag = 0
variable count = 0
variable ns_x = 0
variable ns_y = 0
variable nd = 0
variable found = 0
variable dx, dy
variable p,q
variable last_i=x
variable last_j=y
duplicate/o skeleton,im
Make/O/N=(DimSize(im, 0), DimSize(im, 1)) path
path = 0
path[x][y] = 1
im[x][y]=0
//find one of its neighbors and make it the start coordinate
	for(p=StartX-1;p< StartX+2;p++)
	for(q= StartY-1;q< StartY+2;q++)
	
			if(CheckEndpointType(im,p,q) !=0)
				variable side=checkside(skeleton,startx,starty,p,q)
				if(side==0)
					ns_x+=1
				elseif(side==1)
					ns_y+=1
				elseif(side==2)
					nd+=1
				endif
				variable 	next_i=p
				variable 	next_j=q
			endif
	endfor
	endfor

x=next_i
y=next_j
path[x][y] = 1
variable m=0
variable type=0

Do
    found = 0
    if (isBranch(skeleton,x,y))
        break
    endif
	
    // Loop through all neighbors, checking that the pixels are uncounted to avoid double-counting. Loop through diagonals last.
    for (dx = -1; dx <= 1; dx += 1)
        for (dy = -1; dy <= 1; dy += 1)
            if (dx == 0 && dy == 0)
                continue
            endif

            // Check for out-of-bounds indices
            if (x + dx < 0 || x + dx >= DimSize(im, 0) || y + dy < 0 || y + dy >= DimSize(im, 1))
                continue
            endif

            if (im[x + dx][y + dy] == 1 && path[x + dx][y + dy] == 0)
                path[x + dx][y + dy] = 1
                n += 1

                if (abs(dx) + abs(dy) == 1) // side neighbors
                    if (dx != 0) // horizontal neighbor
                        ns_x += 1
                    else // vertical neighbor
                        ns_y += 1
                    endif
                else // diagonal neighbors
                    n_diag += 1
                endif

                next_i = x + dx
                next_j = y + dy
                count += 1
                found = 1
                break
            endif
        endfor
        if (found)
            break
        endif
    endfor

    if (!found) // No more neighbors found
        break
    endif

    // Re-initialize to the next pixel
    last_i=x
    last_j=y
    x = next_i
    y = next_j

    nd += n_diag
    // Re-initialize counting parameters
    n = 0
    n_sides = 0
    n_diag = 0
    
While (1)

variable delta_x=dimdelta(im,0)

variable delta_y=dimdelta(im,1)

variable CL_x=ns_x*delta_x

variable CL_y=ns_y*delta_y

variable CL_diag=nd*sqrt(delta_x^2+delta_y^2)

variable Contour_length = CL_x + CL_y + CL_diag
variable numpix=ns_x+ns_y+nd
make/o/n=6 TraceResult
TraceResult[0]= Contour_Length
TraceResult[1]=numpix
TraceResult[2]=ns_x
TraceResult[3]=ns_y
TraceResult[4]=nd
TraceResult[5]=0

variable CheckTrace=CheckTraceResult(TraceResult,locations)
if(checkTrace==1)
	TraceResult[5]=1
else
	TraceResult[5]=0
endif
return TraceResult

end


threadsafe Function CheckTraceResult(TraceResult,locations)
wave TraceResult,locations
variable check1,check2
variable numpix1=dimsize(locations,0)
variable numpix2=TraceResult[1]
variable xtotal=TraceResult[2]
variable ytotal=TraceResult[3]
variable dtotal=TraceResult[4]
variable numpix3=xtotal+ytotal+dtotal
if(numpix1==numpix2)
	check1=1
else
	check1=0
endif
if(numpix1==numpix3)
	check2=1
else
	check2=0
endif
if(check1==1&& check2 ==1)
	return 1
else
	return 0
endif

end

threadsafe Function/WAVE TraceComplex(im,locations)
wave im
wave locations
variable n
make/o/n=(dimsize(im,0),dimsize(im,1)) path
path=0
variable xtotal=0,ytotal=0,dtotal=0,c_len=0
for(n=0;n<dimsize(locations,0);n++)
	variable x=locations[n][0]
	variable y=locations[n][1]
	if(im[x][y]==1 && path[x][y]==0)
		wave result=GetPath2(im,x,y,path)
		xtotal+=result[5]
		ytotal+=result[6]
		dtotal+=result[7]
		c_len+=result[4]
	endif

endfor


variable numpix=xtotal+ytotal+dtotal
make/o/n=6 TraceResult
TraceResult[0]=C_len
TraceResult[1]=numpix
TraceResult[2]=xtotal
TraceResult[3]=ytotal
TraceResult[4]=dtotal
TraceResult[5]=0
variable CheckTrace=CheckTraceResult(TraceResult,locations)
if(checkTrace==1)
	TraceResult[5]=1
else
	TraceResult[5]=0
endif
return TraceResult


end


//****************************************************************************************************//
//************************ 			   IV. Utilities 				*************************************//
//****************************************************************************************************//
Function/Wave SizeUp(mask,scale) //each pixel will be split up to get a higher pixel density 
	wave mask
	
	variable scale
	
	make/O/N=(dimsize(mask,0)*scale,dimsize(mask,1)*scale) BigMask
	BigMask = 0
	variable i,j,ii,jj
	
	//First let's just get the outline
	for(i=0;i<dimsize(Mask,0);i++)
		for(j=0;j<dimsize(Mask,1);j++)
		
			if(mask[i][j] == 1)
				for(ii=0;ii<=scale;ii++)
					for(jj=0;jj<=scale;jj++)
						if(i*scale+ii >= dimsize(BigMask,0) || j*scale + jj >= dimsize(BigMask,1))
							continue
						endif
						BigMask[i*scale+ii][j*scale+jj] = 1
					endfor
				endfor
			endif
			
			if(mask[i][j] == 0)
				for(ii=0;ii<=scale;ii++)
					for(jj=0;jj<=scale;jj++)
						if(i*scale+ii >= dimsize(BigMask,0) || j*scale + jj >= dimsize(BigMask,1))
							continue
						endif
						BigMask[i*scale+ii][j*scale+jj] = 0
					endfor
				endfor
			endif
		
		endfor
	endfor
	variable delta_x= dimdelta(mask,0)/scale
	variable delta_y= dimdelta(mask,1)/scale
	setscale/p x,0,delta_x,"m",bigmask
	setscale/p y,0,delta_y,"m",bigmask
	return BigMask
end


Function HeightSlider(S_Struct) : Slidercontrol
	STRUCT WMSliderAction &S_Struct
	
	If (S_Struct.eventcode==9)
	
		SetDrawLayer/K /W=IMAGE overlay
		
		Wave Map= :H_MaxMap
		Wave ScaleMap= :H_MaxScaleMap
		wave im= :im
		wave preview= :preview
		variable uplim =wavemax(im)
		Variable i,j, limI= Dimsize(map,0), limj=Dimsize(map,1)
			For (i=0;i<limI;i+=1)
			For (j=0;j<limJ;j+=1)
			
			If (im[i][j]>S_Struct.curval)
					preview[i][j]=1
					
				Else
					preview[i][j]=0
			endif		
		Endfor		
 		endfor
	
	modifyimage preview ctab={0,uplim,,0}
	doupdate/W=preview
		
	Endif
	Return 0	
End
	

//general pathfinding function
threadsafe Function/wave GetPath2(im, x, y,path)
wave im,path
variable x, y
path[x][y] = 1
variable n = 0
variable n_sides = 0
variable n_diag = 0
variable next_i = x
variable next_j = y
variable count = 0
variable ns_x = 1
variable ns_y = 0
variable nd = 0
variable found = 0
variable dx, dy


variable re_val=0
Do
    found = 0
    if (isbranch(im,x,y))
        re_val=1
        break
    endif

    // Loop through all neighbors, checking that the pixels are uncounted to avoid double-counting. Loop through diagonals last.
    for (dx = -1; dx <= 1; dx += 1)
        for (dy = -1; dy <= 1; dy += 1)
            if (dx == 0 && dy == 0)
                continue
            endif

            // Check for out-of-bounds indices
            if (x + dx < 0 || x + dx >= DimSize(im, 0) || y + dy < 0 || y + dy >= DimSize(im, 1))
                continue
            endif

            if (im[x + dx][y + dy] == 1 && path[x + dx][y + dy] == 0)
                path[x + dx][y + dy] = 1
                n += 1

                if (abs(dx) + abs(dy) == 1) // side neighbors
                    if (dx != 0) // horizontal neighbor
                        ns_x += 1
                    else // vertical neighbor
                        ns_y += 1
                    endif
                else // diagonal neighbors
                    n_diag += 1
                endif
					 //path[x+dx][y+dy]=1
                next_i = x + dx
                next_j = y + dy
                count += 1
                found = 1
                break
            endif
        endfor
        if (found)
            break
        endif
    endfor

    if (!found) // No more neighbors found
        break
    endif

    // Re-initialize to the next pixel
    x = next_i
    y = next_j

    nd += n_diag
    // Re-initialize counting parameters
    n = 0
    n_sides = 0
    n_diag = 0
    
   

While (1)

killwaves/z  count
variable delta_x=dimdelta(im,0)
variable delta_y=dimdelta(im,1)
variable CL_x=ns_x*delta_x
variable CL_y=ns_y*delta_y
variable CL_diag=nd*sqrt(delta_x^2+delta_y^2)

variable Contour_length = CL_x + CL_y + CL_diag
variable numpix=ns_x+ns_y+nd

make/o/n=8 result
result[0]=re_val
result[1]=numpix
result[2]=next_i
result[3]=next_j
result[4]=contour_length
result[5]=ns_x
result[6]=ns_y
result[7]=nd
return result

end


//adds 0 valued padding to edges of an image
threadsafe Function/WAVE AddPadding(im,numlayers)
wave im
variable numlayers
variable im_x=dimsize(im,0)
variable im_y=dimsize(im,1)
variable new_im_x=(2*numlayers)+im_x
variable new_im_y=(2*numlayers)+im_y
make/o/n=(new_im_x,new_im_y) PaddedIm
variable i,j
for(i=0;i<(new_im_x);i++)
for(j=0;j<(new_im_y);j++)
PaddedIm[i][j]=0
endfor
endfor

variable p=0,q=0
for(i=(numlayers);i<(new_im_x-(numlayers));i++)
for(j=(numlayers);j<(new_im_y-(numlayers));j++)
	p= i-numlayers
	q=j-numlayers
	PaddedIm[i][j]=im[p][q]
endfor
endfor


return PaddedIm

end



//determines if a branched particle also loops
threadsafe Function PretzelOrBranched(endpoints,branches)
wave endpoints,branches
variable numEnds=dimsize(endpoints,0)
variable numBranched=dimsize(branches,0)
if(numEnds>numBranched)
	return 0
else
	return 1
endif
end

Function InteractiveContinue(B_Struct) : ButtonControl
	STRUCT WMButtonAction &B_Struct
	
	If( B_Struct.eventCode==2 )	
		KillWindow/Z IMAGE
	EndIf
	
	NVAR quitvar = :interactivequit
	quitvar = 0
	
	Return 0
End

Function InteractiveQuit(B_Struct) : ButtonControl
	STRUCT WMButtonAction &B_Struct
	
	If( B_Struct.eventCode==2 )	
		KillWindow/Z IMAGE
	EndIf
	
	NVAR quitvar = :interactivequit
	quitvar = 1
	
	Return 0
End
//function to create the 3 x 3 grid surrounding an input coordinate
//used for endpoint identification
threadsafe Function/WAVE GeneratePixMap(Skeleton,i,j)
	wave skeleton
	variable i,j 
	if(i-3<0 || i+3>dimsize(skeleton,0) || j-3<0 || j+3>dimsize(skeleton,1))
		wave paddedim=addpadding(skeleton,5)
		variable p=i+5
		variable q=j+5
		duplicate/o paddedim,pixmap
		variable left=p-1, right=p+1,top=q-1,bot=q+1
	else
	duplicate/o Skeleton PixMap
	
	left= i-1
	right= i+1
	top=j-1
	bot=j+1
	endif	
		
	deletepoints/m=0 right+1,1000,Pixmap
	deletepoints/m=1 Bot+1, 1000, pixmap
	deletepoints/m=0 0, left, pixmap
	deletepoints/m=1 0,top, pixmap
	
	killwaves/z paddedim
	Return pixmap


end

//returns what is stored in the wavenote
threadsafe Function/S GetNote(wavenm,notekey)
	wave wavenm
	string notekey
	return stringbykey(notekey,note(wavenm),":","\r")[0,inf]
End

//counts all one-valued pixels within an image
Function CountPixelsIM(im)
wave im
variable i,j,count=0
	for(i=0;i<dimsize(im,0);i++)
	for(j=0;j<dimsize(im,1);j++)
		if(im[i][j]==1)
			Count+=1
		endif
	endfor
	endfor
	return count
end

//deletes last X point on waves
threadsafe function TrimPoints(w)
wave w
variable size=dimsize(w,0)
deletepoints/m=0 (size-1),1,w
end

//returns a list containing exactly one seed pixel per particle on the skeleton image
function/WAVE SeedbyFill(skeleton,returntype)
wave skeleton
variable returntype
duplicate/o skeleton,im
variable i,j,n=0
make/o/n=(1,2) Seeds
for(i=0;i<dimsize(im,0);i++)
for(j=0;j<dimsize(im,1);j++)
	if(im[i][j]==1)
	insertpoints/m=0 n,1,Seeds
	Seeds[n][0]=i
	Seeds[n][1]=j
	n++
	Flood_fill(im,i,j)
	im[i][j]=1
	endif
endfor
endfor

TrimPoints(Seeds)
if(returnType==0)
	return seeds
else
	return im
endif
end

//0 for x-side, 1 for y-side, 2 for diagonal
threadsafe function CheckSide(im,x1,y1,x2,y2)
wave im 
variable x1,x2,y1,y2
variable side
if(x1==x2)
	side=0
elseif(y1==y2)
	side=1
else
	side=2
endif
return side
end


//returns number corresponding to endpoint shape
threadsafe Function CheckEndpointType(skeleton,x,y)
wave skeleton
variable x,y

DFREF currentDF= $GetDatafolder(1)
	NewDatafolder/S/o :pixmaps
	make/o/n=(3,3) Endtype1
							endtype1[0][0]=0
							endtype1[0][1]=0
							endtype1[0][2]=0
							endtype1[1][0]=1
							endtype1[1][1]=1
							endtype1[1][2]=0
							endtype1[2][0]=0
							endtype1[2][1]=0
							endtype1[2][2]=0
					
	make/o/n=(3,3) Endtype2
							endtype2[0][0]=0
							endtype2[0][1]=0
							endtype2[0][2]=0
							endtype2[1][0]=0
							endtype2[1][1]=1
							endtype2[1][2]=0
							endtype2[2][0]=0
							endtype2[2][1]=1
							endtype2[2][2]=0
	make/o/n=(3,3) Endtype3
							endtype3[0][0]=0
							endtype3[0][1]=0
							endtype3[0][2]=0
							endtype3[1][0]=0
							endtype3[1][1]=1
							endtype3[1][2]=1
							endtype3[2][0]=0
							endtype3[2][1]=0
							endtype3[2][2]=0
	make/o/n=(3,3) Endtype4
							endtype4[0][0]=0
							endtype4[0][1]=1
							endtype4[0][2]=0
							endtype4[1][0]=0
							endtype4[1][1]=1
							endtype4[1][2]=0
							endtype4[2][0]=0
							endtype4[2][1]=0
							endtype4[2][2]=0
	make/o/n=(3,3) Endtype5
							endtype5[0][0]=0
							endtype5[0][1]=0
							endtype5[0][2]=0
							endtype5[1][0]=0
							endtype5[1][1]=1
							endtype5[1][2]=0
							endtype5[2][0]=1
							endtype5[2][1]=0
							endtype5[2][2]=0
	make/o/n=(3,3) Endtype6
							endtype6[0][0]=1
							endtype6[0][1]=0
							endtype6[0][2]=0
							endtype6[1][0]=0
							endtype6[1][1]=1
							endtype6[1][2]=0
							endtype6[2][0]=0
							endtype6[2][1]=0
							endtype6[2][2]=0
	make/o/n=(3,3) Endtype7
							endtype7[0][0]=0
							endtype7[0][1]=0
							endtype7[0][2]=1
							endtype7[1][0]=0
							endtype7[1][1]=1
							endtype7[1][2]=0
							endtype7[2][0]=0
							endtype7[2][1]=0
							endtype7[2][2]=0
	make/o/n=(3,3) Endtype8
							endtype8[0][0]=0
							endtype8[0][1]=0
							endtype8[0][2]=0
							endtype8[1][0]=0
							endtype8[1][1]=1
							endtype8[1][2]=0
							endtype8[2][0]=0
							endtype8[2][1]=0
							endtype8[2][2]=1
	make/o/n=(3,3) Endtype9
							endtype9[0][0]=0
							endtype9[0][1]=0
							endtype9[0][2]=1
							endtype9[1][0]=0
							endtype9[1][1]=1
							endtype9[1][2]=1
							endtype9[2][0]=0
							endtype9[2][1]=0
							endtype9[2][2]=0
	make/o/n=(3,3) Endtype10
							endtype10[0][0]=0
							endtype10[0][1]=0
							endtype10[0][2]=0
							endtype10[1][0]=0
							endtype10[1][1]=1
							endtype10[1][2]=1
							endtype10[2][0]=0
							endtype10[2][1]=0
							endtype10[2][2]=1
	make/o/n=(3,3) Endtype11
							endtype11[0][0]=0
							endtype11[0][1]=1
							endtype11[0][2]=1
							endtype11[1][0]=0
							endtype11[1][1]=1
							endtype11[1][2]=0
							endtype11[2][0]=0
							endtype11[2][1]=0
							endtype11[2][2]=0
	make/o/n=(3,3) Endtype12
							endtype12[0][0]=1
							endtype12[0][1]=1
							endtype12[0][2]=0
							endtype12[1][0]=0
							endtype12[1][1]=1
							endtype12[1][2]=0
							endtype12[2][0]=0
							endtype12[2][1]=0
							endtype12[2][2]=0
	make/o/n=(3,3) Endtype13
							endtype13[0][0]=1
							endtype13[0][1]=0
							endtype13[0][2]=0
							endtype13[1][0]=1
							endtype13[1][1]=1
							endtype13[1][2]=0
							endtype13[2][0]=0
							endtype13[2][1]=0
							endtype13[2][2]=0
	make/o/n=(3,3) Endtype14
							endtype14[0][0]=0
							endtype14[0][1]=0
							endtype14[0][2]=0
							endtype14[1][0]=1
							endtype14[1][1]=1
							endtype14[1][2]=0
							endtype14[2][0]=1
							endtype14[2][1]=0
							endtype14[2][2]=0
	make/o/n=(3,3) Endtype15
							endtype15[0][0]=0
							endtype15[0][1]=0
							endtype15[0][2]=0
							endtype15[1][0]=0
							endtype15[1][1]=1
							endtype15[1][2]=0
							endtype15[2][0]=1
							endtype15[2][1]=1
							endtype15[2][2]=0
	make/o/n=(3,3) Endtype16
							endtype16[0][0]=0
							endtype16[0][1]=0
							endtype16[0][2]=0
							endtype16[1][0]=0
							endtype16[1][1]=1
							endtype16[1][2]=0
							endtype16[2][0]=0
							endtype16[2][1]=1
							endtype16[2][2]=1
							
			variable i,j
	
			setdatafolder currentdf
	
			wave pixmap= GeneratePixMap(Skeleton,x,y)
			variable p,q
			variable end1=0, end2=0, end3=0, end4=0,end5=0,end6=0,end7=0,end8=0,end9=0,end10=0,end11=0,end12=0,end13=0,end14=0,end15=0,end16=0
	
				For(p=0;p<3;p++)
				For(q=0;q<3;q++)
					If(pixmap[p][q]==endtype1[p][q])
						end1+=1
						endif
					If(pixmap[p][q]==endtype2[p][q])
						end2+=1
						endif
					If(pixmap[p][q]==endtype3[p][q])
						end3+=1
						endif
					If(pixmap[p][q]==endtype4[p][q])
						end4+=1
						endif
					If(pixmap[p][q]==endtype5[p][q])
						end5+=1
						endif
					If(pixmap[p][q]==endtype6[p][q])
						end6+=1
						endif
					If(pixmap[p][q]==endtype7[p][q])
						end7+=1
						endif
					If(pixmap[p][q]==endtype8[p][q])
						end8+=1
						endif
					If(pixmap[p][q]==endtype9[p][q])
						end9+=1
						endif
					If(pixmap[p][q]==endtype10[p][q])
						end10+=1
						endif
					If(pixmap[p][q]==endtype11[p][q])
						end11+=1
						endif
					If(pixmap[p][q]==endtype12[p][q])
						end12+=1
						endif
					If(pixmap[p][q]==endtype13[p][q])
						end13+=1
						endif
					If(pixmap[p][q]==endtype14[p][q])
						end14+=1
						endif
					If(pixmap[p][q]==endtype15[p][q])
						end15+=1
						endif
					If(pixmap[p][q]==endtype16[p][q])
						end16+=1
					endif
				Endfor
				endfor
	setDatafolder currentDF		
	killdatafolder/Z pixmaps
	killwaves/z pixmap
				
	if(end1==9)
		return 1
	elseif(end2==9)
		return 2
	elseif(end3==9)
		return 3
	elseif(end4==9)
		return 4
	elseif(end5==9)
		return 5
	elseif(end6==9)
		return 6
	elseif(end7==9)
		return 7
	elseif(end8==9)
		return 8
	elseif(end9==9)
		return 9
	elseif(end10==9)
		return 10
	elseif(end11==9)
		return 11
	elseif(end12==9)
		return 12
	elseif(end13==9)
		return 13
	elseif(end14==9)
		return 14
	elseif(end15==9)
		return 15
	elseif(end16==9)
		return 16
	else
		return 0
	endif
end

// Processes each seed pixel in parallel
Function ProcessAllSeeds(HeightSkeleton,countconfirm,lengthswave,numpix,image,seeds,xlocations,ylocations)
	
	wave seeds,image,numpix,countconfirm,lengthswave,xlocations,ylocations,HeightSkeleton
	variable nthreads=threadprocessorcount
	variable nt=threadgroupcreate(nthreads)
	variable i,j
	for(i=0;i<dimsize(Seeds,0);)
	for(j=0;j<nthreads;j++)
		Threadstart nt,j, WorkerThreadFunction(image,HeightSkeleton,seeds,xlocations,ylocations,i,countconfirm,lengthswave,numpix)
		i=i+1
		if(i>=dimsize(seeds,0))
			break
		endif
	endfor
	do
	variable tgs=threadgroupwait(nt,100)
	while(tgs!=0)
	endfor
	variable ThreadStop=threadgrouprelease(nt)


end


Function Flood_fill(im,x,y)
wave im
variable x,y
	if( x<0 ||y <0 || x>dimsize(im,0) || y>dimsize(im,1) || (x-1) <0 || (y-1) <0 || (x+1)>dimsize(im,0) || (y+1)>dimsize(im,1))
		return 0
	endif
	if(im[x][y]==1)
		im[x][y]=0
		Flood_fill(im, (x+1), y)
		flood_fill(im, (x-1), y)
		flood_fill(im, x, (y+1) )
		flood_fill(im, x, (y-1))
		flood_fill(im, (x+1),(y+1))
		flood_fill(im, (x+1),(y-1))
		flood_fill(im,(x-1),(y+1))
		flood_fill(im,(x-1),(y-1))
	endif
	return 0
end

//worker function to process seeds in parallel
threadsafe Function WorkerThreadFunction(image,HeightSkeleton,seeds,xlocations,ylocations,index,countconfirm,lengthswave,numpix)
wave seeds,image,xlocations,ylocations,countconfirm,lengthswave,numpix,HeightSkeleton
variable index
wave wResult
	variable x1=seeds[index][0]
	variable y1=seeds[index][1]
	wave locations=SquareExpandDFS(image,x1,y1)
	variable n
	for(n=0;n<dimsize(xlocations,0);n++)
	if(n<dimsize(locations,0))
	xlocations[n][index]=locations[n][0]
	ylocations[n][index]=locations[n][1]
	endif
	endfor
	
	wave wResult=TraceParticle(HeightSkeleton,image,locations)
			
	lengthsWave[index]=wResult[0]
	CountConfirm[Index]=wResult[5]
	numpix[index]=wResult[1]
   
   
end

//finds continuous locations by making a square with an expanding radius from the seed pixel
threadsafe Function/WAVE SquareExpandDFS(skeleton,x,y)
wave skeleton
variable x,y
make/o/n=(1,2) ParticleLocations
variable i,n=0
variable r=0

do
	wave square=SquareExpand(skeleton,x,y,r)
	wave FoundLocations=CheckSquarePAth(skeleton,square,x,y)
	if(FoundLocations[0][0]==-1||FoundLocations[0][1]==-1)
		break
	endif
	for(i=0;i<dimsize(FoundLocations,0);i++)
		insertpoints/m=0 n,1,ParticleLocations
		ParticleLocations[n][0]=FoundLocations[i][0]
		ParticleLocations[n][1]=FoundLocations[i][1]
	endfor
	r++
while(1)
trimpoints(Particlelocations)
wave Locations=DeleteRepeats(Particlelocations)
return Locations

end

threadsafe Function/WAVE SquareExpand(im,x,y,r)
wave im
variable x,y,r
variable xlow=x-r
variable xhigh=x+r
variable ylow=y-r
variable yhigh=y+r
wave Square=MakeSquare(im,xlow,xhigh,ylow,yhigh)

return square
end

threadsafe Function/WAVE MakeSquare(skeleton,xlow,xhigh,ylow,yhigh)
wave skeleton
variable xlow,xhigh,ylow,yhigh
make/o/n=(1,3) Square

if(xlow<0)
xlow=0
endif
if(ylow<0)
ylow=0
endif
if(xhigh>=dimsize(skeleton,0))
	xhigh=dimsize(skeleton,0)-1
endif
if(yhigh>=dimsize(skeleton,1))
	yhigh=dimsize(skeleton,1)-1
endif

variable i,j,n=0
for(i=xlow;i<=xhigh;i++)
	insertpoints/m=0 n,1,Square
	Square[n][0]=i
	Square[n][1]=ylow
	Square[n][2]=skeleton[i][ylow]
	n++
endfor
for(j=ylow+1;j<=yhigh;j++)
	insertpoints/m=0 n,1,Square
	Square[n][0]=xhigh
	Square[n][1]=j
	Square[n][2]=skeleton[xhigh][j]
	n++
endfor
for(i=xhigh-1;i>=xlow;i-=1)
	insertpoints/m=0 n,1,Square
	Square[n][0]=i
	Square[n][1]=yhigh
	square[n][2]=skeleton[i][yhigh]
	n++
endfor
for(j=yhigh-1;j>=ylow;j-=1)
	insertpoints/m=0 n,1,Square
	Square[n][0]=xlow
	Square[n][1]=j
	square[n][2]=skeleton[xlow][j]
	n++
endfor
trimpoints(square)
return Square
end

//returns wave containing locations within square connected to a target pixel through DFS
threadsafe Function/WAVE CheckSquarePAth(skeleton,square,x,y)
wave skeleton,square
variable x,y
make/o/n=(1,2) FoundLocations
variable n,r=0
variable Connected=0
for(n=0;n<dimsize(square,0);n++)
	if(square[n][2]==1)
		variable x2=square[n][0]
		variable y2=square[n][1]
		if(isContinuousLine(skeleton,x,y,x2,y2)==1)
			Connected+=1
			insertpoints/m=0 r,1,FoundLocations
			FoundLocations[r][0]=x2
			FoundLocations[r][1]=y2
			r++
		endif
	endif
endfor

if(connected==0)
	FoundLocations[0][0]=-1
	FoundLocations[0][1]=-1
else
trimpoints(foundLocations)
endif

return FoundLocations

end

//determines if (x1,y1) is continuous with (x2,y2)
threadsafe Function isContinuousLine(im, x1, y1, x2, y2)
wave im
variable x1, y1, x2, y2

make/o/n=(dimsize(im,0), dimsize(im,1)) visited
visited = 0
return DFS(im, visited, x1, y1, x2, y2)

end

//Depth-First search algorithm
threadsafe Function DFS(im, visited, x, y, target_x, target_y)
    wave im, visited
    variable x, y, target_x, target_y

    if (x < 0 || x >= dimsize(im,0) || y < 0 || y >= dimsize(im,1) || visited[x][y] == 1 || im[x][y] == 0)
        return 0
    endif

    if (x == target_x && y == target_y)
        return 1
    endif

    visited[x][y] = 1

    variable dx, dy
    for (dx = -1; dx <= 1; dx += 1)
        for (dy = -1; dy <= 1; dy += 1)
            if (dx != 0 || dy != 0)
                if (DFS(im, visited, x + dx, y + dy, target_x, target_y) == 1)
                    return 1
                endif
            endif
        endfor
    endfor

    return 0
end

//removes repeated points found with square expand DFS
threadsafe function/wave DeleteRepeats(locations)
wave locations
variable n,r=0
make/o/n=(1,2) newLocations

variable m
variable check=0
for(m=0;m<dimsize(locations,0);m++)
	variable x1=Locations[m][0]
	variable y1=Locations[m][1]
	for(n=0;n<dimsize(locations,0);n++)
		variable x2=locations[n][0]
		variable y2=locations[n][1]
		if(m==n)
			continue
		endif
			if(x1==x2 && y1==y2)
				check=1
			endif
		
	endfor
	if(check==0)
	insertpoints/m=0 r,1,newLocations
	newlocations[r][0]=x1
	newlocations[r][1]=y1
	r++
	elseif(check==1)
		variable t
		variable check2=0
		for(t=0;t<dimsize(NewLocations,0);t++)
			variable x3=newlocations[t][0]
			variable y3=newlocations[t][1]
			if(x1==x3 && y1==y3)
				 check2=1
			endif
			
		endfor
		if(check2==0)
				insertpoints/m=0 r,1,newLocations
				newlocations[r][0]=x1
				newlocations[r][1]=y1
				r++
			endif
	endif
endfor
trimpoints(newlocations)

return newlocations


end


//checks the 8 surrounding pixels and counts 1's
function numNeighbors(skeleton, x, y)
wave skeleton
variable x, y

variable neighbors = 0
variable i, j
if( (x+1) >= dimsize(skeleton,0) || (y+1) >=dimsize(skeleton,1) || (x-1) <0 || (y-1)<0)
	return 0
endif

if(skeleton[x+1][y] == 1)
	neighbors += 1
endif
if(skeleton[x-1][y] == 1)
	neighbors += 1
endif
if(skeleton[x][y+1] == 1)
	neighbors += 1
endif
if(skeleton[x][y-1] == 1)
	neighbors += 1
endif
if(skeleton[x-1][y+1] == 1)
	neighbors += 1
endif
if (skeleton[x+1][y+1] == 1)
	neighbors += 1
endif
if(skeleton[x-1][y-1] == 1)
	neighbors += 1
endif
if(skeleton[x+1][y-1] == 1)
	neighbors += 1
endif

return neighbors

end

//****************************************************************************************************//
//************************ 			 V. Noise Cleaning Functions 				***************************//
//****************************************************************************************************//

//removes 1-valued pixels with no neighbors from the skeleton image
Function ClearSinglePixelNoise(im)
wave im
variable i,j

for(i=0;i<dimsize(im,0);i++)
for(j=0;j<dimsize(im,1);j++)
	if(im[i][j]==1 && numneighbors(im,i,j)==0)
	im[i][j]=0
	endif
endfor
endfor

end

//Removes edge particles from skeleton image within n pixels of the boundary
Function/WAVE CleanSkeletonEdges(skeleton,n)
wave skeleton
variable n
variable i,j

duplicate/o skeleton,newSkeleton

for(i=0;i<=n;i++)
for(j=0;j<dimsize(skeleton,1);j++)
	if(newSkeleton[i][j]==1)
		Flood_fill(newSkeleton,i,j)
	endif
endfor
endfor

for(i=0;i<dimsize(skeleton,0);i++)
for(j=0;j<=n;j++)
	if(newSkeleton[i][j]==1)
		Flood_fill(newSkeleton,i,j)
	endif
endfor
endfor

variable limX=dimsize(skeleton,0)-n
for(i=limX;i<(dimsize(skeleton,0));i++)
for(j=0;j<dimsize(skeleton,1);j++)
	if(newSkeleton[i][j]==1)
		Flood_fill(newSkeleton,i,j)
	endif
endfor
endfor

variable limY=dimsize(skeleton,1)-n
for(i=0;i<dimsize(skeleton,0);i++)
for(j=limY;j<dimsize(skeleton,1);j++)
	if(newSkeleton[i][j]==1)
		Flood_fill(newSkeleton,i,j)
	endif
endfor
endfor

return newSkeleton

end