/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../include/common.h"

/*
 * Transform the column stored string from AOS to SOA.
 * Currently the column must be stored in an uncompressed format.
 */

int main(int argc, char **argv){
	int fd;
	char * buf;

	if (argc != 3){
		printf("./soa columnName columnWidth\n");
		exit(-1);
	}

	int columnWidth = atoi(argv[2]);

	fd = open(argv[1], O_RDWR);
	
	struct columnHeader header;

	read(fd, &header, sizeof(struct columnHeader));

	if (header.format != UNCOMPRESSED){
		printf("Not support uncompressed column yet!");
		exit(-1);
	}

	int blockTotal = header.blockTotal;

	long offset = 0;
	long tupleOffset = 0;
	for(int i=0;i<blockTotal;i++){
		offset = i* sizeof(struct columnHeader) + tupleOffset * columnWidth;

		lseek(fd, offset, SEEK_SET);

		offset += sizeof(struct columnHeader);
		read(fd, &header, sizeof(struct columnHeader));
		int blockSize = header.blockSize;
		int tupleNum = header.tupleNum;
		buf = (char *) malloc(blockSize);

		char  * tmp = (char *) malloc(columnWidth +1);
		for(int j=0;j<tupleNum;j++){
			memset(tmp, 0, columnWidth + 1);
			read(fd,tmp, columnWidth);

			for(int k=0;k<columnWidth;k++){
				int pos = k*tupleNum + j; 
				buf[pos] = tmp[k];
			}
		}

		lseek(fd,offset,SEEK_SET);
		write(fd,buf,blockSize);
		free(tmp);
		free(buf);
		tupleOffset += tupleNum;
	}

	close(fd);
	return 0;
}
