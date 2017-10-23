
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

function generate2() { //Added by mirukuma 
  const model = new KerasJS.Model({ //--Added by mirukuma
    filepaths: {
      model: './data/generator.json',
      weights: './data/generator_weights.buf',
      metadata: './data/generator_metadata.json'
    },
    gpu: true
  })
  console.log(model)

  model.ready()
    .then(() => {
      // input data object keyed by names of the input layers
      // or `input` for Sequential models
      // values are the flattened Float32Array data
      // (input tensor shapes are specified in the model config)
      var noise= new Float32Array(100)
      for(i=0; i<100; i=i+1){
        noise[i]=Math.random()*2-1
      }
      const inputData = {
        'input': noise
      }

      console.log(inputData)
      // make predictions
      return model.predict(inputData)
    })
    .then(outputData => {
      // outputData is an object keyed by names of the output layers
      // or `output` for Sequential models
      // e.g.,
      // outputData['fc1000']
      console.log(outputData)
    })
    .catch(err => {
      console.log(err)
    })

}
