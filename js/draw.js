CanvasRenderingContext2D.prototype.drawImageByDada=function($Array,$width,$height,$find,$px,$x,$y){
	let $dy=$y;
	let $idx=$find;
	for($hc=0;$hc<$height;$hc++){
		let $dx=$x;
		for($wc=0;$wc<$width;$wc++){
			let $color=$Array[$idx];
			$color.forEach((e,i,a)=>{a[i]=e*127.5+127.5});
			this.drawFillBox($dx,$dy,$px,`rgb(${$color[0]},${$color[1]},${$color[2]})`);
			$dx+=$px;
			$idx++;
		}
		$dy+=$px;
	}
}
