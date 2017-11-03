var d=document;
var m=Math;

var w,h;
var canv,cont;
var generatedData; // generate2() の戻り値を格納する変数

window.onload=()=>{
	canv=d.getElementById('output');
	cont=canv.getContext('2d');
	
	w=d.getElementById('w');
	h=d.getElementById('h');
	
	canv.width=164;
	canv.height=200;
	
	textRoll('-Generate ultimate cute girls-',d.getElementById('display'),50
		 ,d.getElementById('shel'),'<h1>-Generate <span class="text">ultimate cute</span> girls.-</h1>');
}

function textRoll(text,target,interval,subtarget,afterInput){
	let cnt=0;
	let roll=setInterval(()=>{
		cnt++;
		target.innerText=text.substr(0,cnt);
		if(cnt>text.length){
			clearInterval(roll);
			if(afterInput!=void(0))subtarget.innerHTML=afterInput;
		}
	},interval);
}

function setImgSize(){
	wid=~~w.value;
	hig=~~h.value;
	console.log([w,h]); // test
}
