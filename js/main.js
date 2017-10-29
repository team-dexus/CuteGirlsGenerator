var d=document;
var m=Math;

var canv,cont;

window.onload=()=>{
	canv=d.getElementById('output');
	cont=canv.getContext('2d');
	
	canv.width=128;
	canv.height=128;
	
	textRoll('-Generate ultimate cute girls-',d.getElementById('shel'),50
		 ,'<h1>-Generate <span class="text">ultimate cute</span> girls.-</h1>');
}

function textRoll(text,target,interval,afterInput){
	let cnt=0;
	let roll=setInterval(()=>{
		cnt++;
		target.innerText=text.substr(0,cnt);
		if(cnt>text.length){
			clearInterval(roll);
			if(afterInput!=void(0))target.innerHTML=afterInput;
		}
	},interval);
}
