function modelEstimator(){
this.selectSliceCode=function(X, start, stop, axis){
    // fix starting behavior
    if(start == null){
        start = 0
    }
    
    // test number of axes
    var axes = 0
    var Xv = X
    
    while(Xv instanceof Array){
        axes += 1
        Xv = Xv[0]
    }
    
    var make_array = function(x, pos_axis=0){
        var neg_axis = pos_axis - axes 
        if((pos_axis == axis) || (neg_axis == axis)){
            var x_new = null
            
            if(stop == null && start == null){
                x_new = x
            }else{
                // Make slice
                if(stop == null){
                    x_new = x.slice(start) // takes all elements until the last one
                }else{
                    x_new = x.slice(start, stop)
                }
            }
            
            return x_new
        }
        var result = []
        var clevel = pos_axis+1
        for(var i=0; i<x.length; i++){
            result.push(make_array(x[i], clevel))
        }
        return result
    }
    
    return make_array(X)
}
this.transform__features__padsubsequence=function(X){
    var length = 10
    var step = 1
    var n_features = 4096
    X = this.padSubsequenceCode(X, length, step, n_features)
    return X
}
this.transform__features=function(X){
    X = this.transform__features__padsubsequence(X)
    X = this.transform__features__spectrumtransform(X)
    X = this.transform__features__selectslice(X)
    X = this.transform__features__flattenshape(X)
    return X
}
this.padSubsequenceCode=function(X, length, step, n_features){
    var result = []

    // instance enumeration
    for(var i=0; i<X.length; i++){
        var x = X[i];
        var x_new = []
        // pad with zeros
        for(var j=0; j<length; j++){
            if(x.length-j > 0){
                x_new.unshift(x[x.length-j-1])
            }else{
                var zeros = []
                for(var k=0; k<n_features; k++){
                    zeros.push(0)
                }
                x_new.unshift(zeros)
            }
        }
        var x_step = []
        for(var j=0; j<length; j++){
            if(j % step == 0){
                x_step.push(x_new[j])
            }
        }
        result.push(x_step)
    }

    return result
}
this.transform__model=function(X){
    return X
}
this.transform__features__selectslice=function(X){
    var start = 0
    var stop = 256
    var axis = -1
    X = this.selectSliceCode(X, start, stop, axis)
    return X
}
this.postprocess=function(X){
    var classes_ = ["Artifact", "Extra heart sounds", "Murmur", "Normal"]
    X = this.labelBinarizerCode(X, classes_)
    return X
}
this.transform=function(X){
    X = this.transform__features(X)
    return X
}
this.flattenShapeCode=function(X){
    var result = []
    
    // test number of axes
    var axes = 0
    var Xv = X
    
    while(Xv instanceof Array){
        axes += 1
        Xv = Xv[0]
    }
    
    var myfunc = function(arr, axis=0){
        if(axis==axes-1){
            return arr
        }
        
        var result = []
        for(var i=0; i<arr.length; i++){
            result.push(myfunc(arr[i], axis+1))
        }
        
        // flatten up to first dimension
        if(axis > 0){
            result = [].concat.apply([], result)
        }
        return result
    }

    return myfunc(X)
}
this.labelBinarizerCode=function(X, classes_){
    result = []
    
    for(var i=0; i<X.length; i++){
        var y = X[i];
        var max_i = 0
        // find the maximum index in array
        for(var j = 0; j<y.length; j++){
            if(y[max_i] < y[j]){
                max_i = j
            }
        }
        result.push(classes_[max_i])
    }
    return result
}
this.transform__features__spectrumtransform=function(X){
    var axis = -1
    X = this.spectrumTransformCode(X, axis)
    return X
}
this.load=function(config, onloaded){
    this.model = new KerasJS.Model({
        filepath: config['weights'],
        gpu: false
    })
    this.model.ready().then(()=>{
        onloaded()
    })
}
this.spectrumTransformCode=function(X, axis){
    var result = []

    // instance enumeration
    for(var i=0; i<X.length; i++){
        var x = X[i];
        var x_new = []
        // time step enumeration
        for(var t=0; t<x.length; t++){
            var t_size = x[t].length

            var fft = new FFT(x[t].length, 1);
            fft.forward(x[t]);

            // feature enumeration
            var t_new = []

            for(var j=0; j<fft.real.length; j++){
                var a = fft.real[j]
                var b = fft.imag[j]
                t_new.push(Math.sqrt(a*a+b*b))
            }

            x_new.push(t_new)
        }
        result.push(x_new)

    }

    return result
}
this.predict=function(X, onpredict){
    var X = this.transform(X)
    var X_new = []
    var this_model = this.model
    var this_this = this
    
    var recurr = function(outputData){
        if(outputData != null){
            X_new.push(outputData['Output_1'])
        }
        
        if(X_new.length >= X.length){
            X_new = this_this.postprocess(X_new)
            onpredict(X_new, null)
        }else{
            var x = X[X_new.length]
            x = {"Input_1": new Float32Array(x)}
            this_model.predict(x).then(recurr).catch(err => {
                onpredict(null, err)
            })
        }
    }
    
    recurr(null)
}
this.transform__features__flattenshape=function(X){
    X = this.flattenShapeCode(X)
    return X
}
}