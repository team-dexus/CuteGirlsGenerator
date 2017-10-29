CanvasRenderingContext2D.prototype.drawImageByDada=function($Array,$width,$height,$find,$px,$x,$y){
	let $dy=$y;
	let $idx=$find;
	for($hc=0;$hc<$height;$hc++){
		let $dx=$x;
		for($wc=0;$wc<$width;$wc++){
			let R=$Array[$idx+0]*127.5+127.5;
			let G=$Array[$idx+1]*127.5+127.5;
			let B=$Array[$idx+2]*127.5+127.5;
			R=~~R+',';G=~~G+',';B=~~B+'';
			if($idx==0)console.log([R,G,B]);
			this.drawFillBox($dx,$dy,$px+1,`rgb(${R+G+B})`);
			$dx+=$px;
			$idx+=3;
		}
		$dy+=$px;
	}
	console.log($idx);
}

function drawGeneratedImage(){
      cont.clearRect(0,0,canv.width,canv.height);
      cont.drawImageByDada(generatedData,24,40,0,7,0,0); // test code
}
