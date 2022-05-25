# Dimentions
>nPoints_XY(1203713)  
nPoints_XYZ(1203713)  
nMetLines(24)  
nGroups(1)  
nDatasets(6)  
MetLineStringLength(11)  
DatasetNameStringLength(34)  
GroupStringLength(20)  
PointNameStringLength(34)  
OutputOptionsStringLength(14)  
OutputOptionsUsed(3)   
OutputPointTypesLength(16)  
NumberOfTypesOfOutputPoint(5)  

# Variables
## XY
>float32 PointX_XY(nPoints_XY)    
float32 PointY_XY(nPoints_XY)   
|S1 PointName_XY(nPoints_XY, PointNameStringLength)   
float32 PointX_XYZ(nPoints_XYZ)   
float32 PointY_XYZ(nPoints_XYZ)   
float32 PointZ_XYZ(nPoints_XYZ)
|S1 PointName_XYZ(nPoints_XYZ, PointNameStringLength)
- XY is the same as the XYZ with Z=1
- resolution is 200x200m
- PointNames are all 'Grid, grid'
- where is (0, 0)?
## Met
>|S1 Met_Line(nMetLines, MetLineStringLength)   
- Units: year_day_hour, like '2022_002_21'

>float32 Met_Freq(nMetLines)   
- all are 1.

>float32 Met_UAt10m(nMetLines)   
- Units: m/s
- example:  
[3.780007  2.500009  3.990004  6.54      7.0996037 7.649593  8.109708
 7.71964   7.519601  5.069602  3.59028   4.039662  4.169536  4.500002
 4.230005  3.2594864 2.7998729 3.4496448 3.3298786 3.3699667 3.1200945
 2.7095635 3.330355  4.060108 ]

>float32 Met_PHI(nMetLines)  
- dgree
- example:  
[103.8  97.1  95.9  77.3  71.9  72.   75.   78.6  80.1  71.3  72.3  80.1
  86.3  92.   92.5  99.7 123.7 153.5 200.  183.7 174.6  93.9  90.8  92.8]

>float32 Met_H_over_LMO(nMetLines)  
- example:  
[ 0.43638772  0.4274883   0.39823866  0.05592077 -0.11760374 -0.2310089
 -0.34271598 -0.4624248  -0.5481426  -1.0096766  -2.2684886  -1.1935143
 -0.11627784  0.318594    0.48602146 -4.505068   -4.6965814  -2.2882512
 -6.3754067  -5.7775097  -6.7984743  -0.35236338 -1.1351125  -0.8154919 ]

## Dataset
>|S1 Group(nGroups, GroupStringLength)  
- 'All sources'

>|S1 DatasetNames(nDatasets, DatasetNameStringLength)   
- O3_conc_1hour, O2_conc_1hour， NOx_conc_1hour， PM2.5_conc_1hour， PM10_conc_1hour， SO2_conc_1hour
- NOx? NO1 or NO2?

>|S1 Output_Options_Description(OutputOptionsUsed, OutputOptionsStringLength)
- Concentration, Dry deposition, Wet deposition

>int32 Output_Options_Used(OutputOptionsUsed)  
- example:  
[1 0 0]

>|S1 Output_Points_Type(NumberOfTypesOfOutputPoint, OutputPointTypesLength)  
- Grid Horiz, Grid All, Specified Points, Grid Nested, Grid Intelligent
- How to understand thest 5 type?

>int32 Number_Of_Output_Points_Of_Each_Type(NumberOfTypesOfOutputPoint)
- example:  
[  73200   73200      18       0 1130495]
- Grid Horiz and Grid All are overlap?
- 305x240 => 73200
- 18 stations

>float32 Dataset1(nMetLines, nGroups, nPoints_XYZ)  
float32 Dataset2(nMetLines, nGroups, nPoints_XYZ)  
float32 Dataset3(nMetLines, nGroups, nPoints_XYZ)   
float32 Dataset4(nMetLines, nGroups, nPoints_XYZ)  
float32 Dataset5(nMetLines, nGroups, nPoints_XYZ)   
float32 Dataset6(nMetLines, nGroups, nPoints_XYZ)   
