TITLE: Taxicab Trip Information

ABSTRACT:  DC Taxi trips from 2019. The zip file contains zip files of pipe (|) delimeted text files and CSV for trips by month. The record counts are:

Month			Count
01/2019     	  589,475
02/2019     	  667,546
03/2019     	  902,382
04/2019     	  853,950
05/2019     	  865,374
06/2019    	  800,030
07/2019           699,064
08/2019           581,921
09/2019           700,043
10/2019           794,546
11/2019           623,990
12/2019           574,400


DISCLAIMER: This work is licensed under a Creative Commons Attribution 4.0 International License.

CATEGORY: Public Service	

PROVIDER: Office of the Chief Technology Officer (OCTO)

ORIGINATOR: Department of For-Hire Vehicles (DFHV)

PROCESS DESCRIPTION: DFHV provided OCTO with a taxicab trip text file representing trips from January 2015 to June 2019.  OCTO processed the data to assign a block locations to pick up and drop off locations.  The blocks were assigned using the original pick up, drop off lat/long coordinates and searching for the block locations in the DC Master Address Repository (radius tolerance of 250 meters and less).  The pick and drop off times were also rounded to the nearest hour.

In addition, the pick up and drop off locations were assigned to an airport using locator polygons for Reagan, BWI, and Dulles.  These polygons generally followed the visual borders of these airports.

For more information, go to http://dfhv.dc.gov

TABLE STRUCTURE:	 

COLUMN_NAME		DATA_TYPE		DESCRIPTION
OBJECTID			NUMBER(9)		Table Unique Identifier	
TRIPTYPE			VARCHAR2(255)	Type of Taxi Trip
PROVIDERNAME		VARCHAR2(255)	Taxi Company that Provided 								trip	
FAREAMOUNT			VARCHAR2(255)	Meter Fare Amount
GRATUITYAMOUNT		VARCHAR2(255)	Tip amount
SURCHARGEAMOUNT		VARCHAR2(255)	Surcharge fee
EXTRAFAREAMOUNT		VARCHAR2(255)	Extra fees
TOLLAMOUNT			VARCHAR2(255)	Toll amount
TOTALAMOUNT		VARCHAR2(255)	Total amount from Meter 								fare, tip, surcharge, 								extras, and tolls. 
PAYMENTTYPE		VARCHAR2(255)	Payment type
ORIGINCITY			VARCHAR2(255)	Pick up location city
ORIGINSTATE		VARCHAR2(255)	Pick up location state
ORIGINZIP			VARCHAR2(255)	Pick up location zip
DESTINATIONCITY		VARCHAR2(255)	Drop off location city
DESTINATIONSTATE		VARCHAR2(255)	Drop off location state
DESTINATIONZIP		VARCHAR2(255)	Drop off location zip
MILEAGE			VARCHAR2(255)	Trip milaege
DURATION			VARCHAR2(255)	Trip time - duration of 								travel
ORIGIN_BLOCK_LATITUDE	NUMBER		Pick up location latitude
ORIGIN_BLOCK_LONGITUDE/NUMBER		Pick up location longitude
ORIGIN_BLOCKNAME		VARCHAR2(255)	Pick up location street 								block name
DESTINATION_BLOCK_LAT	NUMBER		Drop off location latitude
DESTINATION_BLOCK_LONGNUMBER		Drop off location longitude
DESTINATION_BLOCKNAME	VARCHAR2(255)	Drop off location street 								block name
AIRPORT			CHAR(1)		Pick up or drop off 									location is a local airport 							(Y/N)
ORIGINDATETIME_TR	DATE			Pick up date and time	
DESTINATIONDATETIME_TRDATE			Drop off date and time	


Payment Type
1.	Credit
2.	Cash
3.	EHail 
4.	Other (not sure how common this is)
5.	Uber (not sure how common this is)
Trip Type
1.	Ordinal (normal rate)
2.	VoD 
3.	TransportDC (grant program)
4.	TransportDCShared (grant program)
5.	MOVA (grant program)
6.	CFSA (grant program)
7.	NRS (grant program)
8.	NEMT (grant program


