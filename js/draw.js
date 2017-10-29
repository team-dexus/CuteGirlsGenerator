CanvasRenderingContext2D.prototype.drawImageByDada=function($Array,$width,$height,$find,$px,$x,$y){
	let $dy=$y;
	let $idx=$find;
	for($hc=0;$hc<$height;$hc++){
		let $dx=$x;
		for($wc=0;$wc<$width;$wc++){
			let $color=$Array[$idx];
			$color=[$Array[$idx+0],$Array[$idx+1],$Array[$idx+2]];
			$color.forEach((e,i)=>{$color[i]=e*127.5+127.5});
			this.drawFillBox($dx,$dy,$px,`rgb(${~~$color[0]-70},${~~$color[1]},${~~$color[2]},)`);
			$dx+=$px;
			$idx+=3;
		}
		$dy+=$px;
	}
	console.log($idx);
}
