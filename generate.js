
function generate(){ //Added by mirukuma ,Changed by JPNYKW ,Changed by mirukuma ,Changed by JPNYKW
  var pic=document.getElementById('GirlPic');
  var item=['MG7.png','MG57.png','code100.png']
  var text=document.getElementById('words').value;
  var arrayRes=document.getElementById('ArrayResult');
  var result='===WORDS===<br>';
  tag(text).forEach(function(e,i,a){result+=`<span id=res${i}>${e}</span><br>`});
  document.getElementById('ShowText').textContent='Generation result in '+text;
  document.getElementById('ArrayRes').innerHTML=`${result}=========`;
  pic.src=item[~~(Math.random()*3)];


}

function generate2(){ //Added by mirukuma ,Changed by JPNYKW ,Changed by mirukuma ,Changed by JPNYKW
  const model = new KerasJS.Model({ //--Added by mirukuma
    filepaths: {
      model: '/data/model.json',
      weights: '/data/generator2_weights.buf',
      metadata: '/data/generator2_metadata.json'
    },
    gpu: true
  })


  inputData = new Float32Array(100)
  for(var i=0; i<100; i++){
    inputData[i]=Math.random();
  }
  console.log("life!")

  out=model.ready().then(() => {
    model.predict(inputData).then(outputData => {
          var output_tensor = outputData.float_m1p1;
          console.log("sex")
          var imgarray = new Uint8Array(output_tensor.data.map(function (x) {
              return (x + 1) / 2 * 255;
          }));
          var imageData = image2Darray(imgarray, output_tensor.shape[0], output_tensor.shape[1]);
          console.log("is")
          var canvas = (typeof(canvas_or_id) === 'string') ? $('#' + canvas_or_id)[0] : canvas_or_id;
          if (canvas) {
              var context = canvas.getContext('2d');
              context.putImageData(imageData, 0, 0);
          }

        return imageData;
       //ここにoutputを書く。
    })
  })
  return out;
}
