var inText='Generate ultimate cute girls.-';
var s='-';
var c=0;

tag=str=>str.split(/[\　\ ]+/ig); // search words open

function init(){
	var display=document.getElementById('display');
	var shel=document.getElementById('shel');
   }

function main(){
	if(c<inText.length){
	s+=inText.substr(c,1);
       	display.innerText=s; // animation of text motion
       	c++;
}else if(c==inText.length){
       	shel.innerHTML='<h1>-Generate <span class=\"text\">ultimate cute</span> girls.-</h1>'; // set html tag
       	c++
	}
}

setInterval(main,70); // call main function in 70/ms
