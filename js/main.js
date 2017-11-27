var download,onGenerated=false;
var fileIndex=-1;
var d=document;
var c=console;
var w=window;
var m=Math;

var w,h;
var canv,cont;
var generatedData; // generate2() の戻り値を格納する変数

window.onload=()=>{
	download=d.getElementById("download");
	canv=d.getElementById('output');
	cont=canv.getContext('2d');

	w=d.getElementById('w');
	h=d.getElementById('h');

	canv.width=168;
	canv.height=280;

	textRoll('-Generate ultimate cute girls-',d.getElementById('display'),50
		 ,d.getElementById('shel'),'<h1 class="logo">-Generate <span class="text">ultimate cute</span> girls.-</h1>');
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

function updateName(){
	fileIndex++;
	download.download=`PMG_${fileIndex}.png`;
}
