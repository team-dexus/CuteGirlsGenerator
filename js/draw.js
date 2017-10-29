CanvasRenderingContext2D.prototype.drawImageByDada=function($Array,$width,$height,$px,$x,$y){
	let $dy=$y;
	for($hc=0;$hc<$height;$hc++){
		let $dx=$x;
		for($wc=0;$wc<$width;$wc++){
			let $color=$Array[$hc][$wc];
			$color.forEach((e,i,a)=>{a[i]=e*127.5+127.5});
			this.drawFillBox($dx,$dy,$px,`rgb(${$color[0]},${$color[1]},${$color[2]})`);
			$dx+=$px;
		}
		$dy+=$px;
	}
}