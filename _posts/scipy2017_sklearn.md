

```python
from sklearn.datasets import load_iris
iris = load_iris()
```


```python
iris.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])




```python
n_samples, n_features = iris.data.shape
```


```python
print('Number of samples', n_samples)
```

    Number of samples 150
    


```python
print('Number of features', n_features)
```

    Number of features 4
    


```python
print(iris.data[0])
```

    [ 5.1  3.5  1.4  0.2]
    


```python
print(iris.data.shape)
```

    (150, 4)
    


```python
print(iris.target.shape)
```

    (150,)
    


```python
print(iris.target)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]
    


```python
import numpy as np
```


```python
np.bincount(iris.target)
```




    array([50, 50, 50], dtype=int64)




```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
x_index = 3
colors = ['#7CAA2D', '#CB6318', '#1E656D']

for label, color in zip(range(len(iris.target_names)), colors):
    plt.hist(iris.data[iris.target==label, x_index],
             label = iris.target_names[label],
             color = color)
plt.xlabel(iris.feature_names[x_index])
plt.legend(loc= 'upper right')
plt.show()
```


![png](output_12_0.png)



```python
x_index = 1
y_index = 3

colors = ['#7CAA2D', '#CB6318', '#1E656D']

for label, color in zip(range(len(iris.target_names)), colors):
    plt.scatter(iris.data[iris.target==label, x_index],
                iris.data[iris.target==label, y_index],
             label = iris.target_names[label],
             c = color)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.legend(loc= 'upper left')
plt.show()
```


![png](output_13_0.png)



```python
import pandas as pd
```


```python
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
```


```python
iris_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.plotting.scatter_matrix(iris_df, c= iris.target, figsize=(8, 8));
```


![png](output_17_0.png)



```python
iris = sns.load_dataset("iris")
g = sns.pairplot(iris, hue = "species")
```


![png](output_18_0.png)



```python
from sklearn.datasets import load_digits
digits = load_digits()
```


```python
digits.keys()
```




    dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])




```python
n_sam, n_feat = digits.data.shape
```


```python
print(digits.data.shape)
print(digits.images.shape)
```

    (1797, 64)
    (1797, 8, 8)
    


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
x,y = iris.data, iris.target
```


```python
classifier = KNeighborsClassifier()
```


```python
y
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5, test_size = 0.5, random_state = 123)

print("Labels for training and testing data")
print(y_train)
print(y_test)
```

    Labels for training and testing data
    [1 1 0 2 2 0 0 1 1 2 0 0 1 0 1 2 0 2 0 0 1 0 0 1 2 1 1 1 0 0 1 2 0 0 1 1 1
     2 1 1 1 2 0 0 1 2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 2 1 1 2 2 1 0 1 1 2
     2]
    [1 2 2 1 0 2 1 0 0 1 2 0 1 2 2 2 0 0 1 0 0 2 0 2 0 0 0 2 2 0 2 2 0 0 1 1 2
     0 0 1 1 0 2 2 2 2 2 1 0 0 2 0 0 1 1 1 1 2 1 2 0 2 1 0 0 2 1 2 2 0 1 1 2 0
     2]
    


```python
print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(y_train) / float(len(y_train)) * 100.0)
print('Test:', np.bincount(y_test) / float(len(y_test)) * 100.0)
```

    All: [ 33.33333333  33.33333333  33.33333333]
    Training: [ 30.66666667  40.          29.33333333]
    Test: [ 36.          26.66666667  37.33333333]
    


```python
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   train_size = 0.5, test_size = 0.5, random_state = 123, stratify = y)
print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(y_train) / float(len(y_train)) * 100.0)
print('Test:', np.bincount(y_test) / float(len(y_test)) * 100.0)
```

    All: [ 33.33333333  33.33333333  33.33333333]
    Training: [ 33.33333333  33.33333333  33.33333333]
    Test: [ 33.33333333  33.33333333  33.33333333]
    


```python
from sklearn.datasets import make_blobs
```


```python
x, y = make_blobs(centers=2, random_state=0, cluster_std= 1)
```


```python
print('x - n_samples x n_features', x.shape)
print('y - n_samples :', y.shape)
```

    x - n_samples x n_features (100, 2)
    y - n_samples : (100,)
    


```python
print('\n first 5 samples: \n', x[:5, :])
print('\n first 5 labels\n', y[:5])
```

    
     first 5 samples: 
     [[ 4.21850347  2.23419161]
     [ 0.90779887  0.45984362]
     [-0.27652528  5.08127768]
     [ 0.08848433  2.32299086]
     [ 3.24329731  1.21460627]]
    
     first 5 labels
     [1 1 0 0 1]
    


```python
plt.scatter(x[y == 0, 0], x[y == 0, 1],
           c='blue', s=40, label= 'o')
plt.scatter(x[y == 1, 0], x[y == 1, 1],
           c='red', s=40, label = '1', marker = 's')
```




    <matplotlib.collections.PathCollection at 0x14cae564630>




![png](output_35_1.png)



```python
import seaborn as sns
```

### Set style of scatterplot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

### Create scatterplot of dataframe
sns.lmplot('x', # Horizontal axis
           'y', # Vertical axis
           data=df, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="z", # Set color
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size

### Set title
plt.title('Histogram of IQ')

### Set x-axis label
plt.xlabel('Time')

### Set y-axis label
plt.ylabel('Deaths')


```python
sns.regplot(x[y==0, 0], x[y==0,1], color = 'Blue', marker='o')
sns.regplot(x[y==1, 0], x[y==1,1], color = 'Red', marker='s')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14cae7f04e0>




![png](output_38_1.png)



```python
from bokeh.charts import Scatter, output_notebook, show
output_notebook()
```



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="0270fad7-29dd-44c4-9313-6c3f1497de7d">Loading BokehJS ...</span>
    </div>





```python
x_df = pd.DataFrame(x, columns=['A','B'])
y_df = pd.DataFrame(y, columns=['C'])
```


```python
together = pd.concat([x_df,y_df], axis=1)
```


```python
together.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.218503</td>
      <td>2.234192</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.907799</td>
      <td>0.459844</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.276525</td>
      <td>5.081278</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.088484</td>
      <td>2.322991</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.243297</td>
      <td>1.214606</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
p = Scatter(data=together, x = 'A', y = 'B', color='C')
```


```python
show(p)
```

    C:\Users\surhu\Anaconda3\lib\site-packages\bokeh\core\json_encoder.py:73: FutureWarning: pandas.tslib is deprecated and will be removed in a future version.
    You can access Timestamp as pandas.Timestamp
      if pd and isinstance(obj, pd.tslib.Timestamp):
    




    <div class="bk-root">
        <div class="bk-plotdiv" id="f4f7048c-d2e2-4b58-86d4-06a2a4f056a0"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        var el = document.getElementById("f4f7048c-d2e2-4b58-86d4-06a2a4f056a0");
        el.textContent = "BokehJS " + Bokeh.version + " successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("f4f7048c-d2e2-4b58-86d4-06a2a4f056a0");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'f4f7048c-d2e2-4b58-86d4-06a2a4f056a0' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"1ff71593-dc0c-44f0-85a1-ded984c09026":{"roots":{"references":[{"attributes":{"callback":null,"column_names":["x_values","y_values"],"data":{"C":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"chart_index":[{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0},{"C":0}],"x_values":{"__ndarray__":"eMMkGJey0b+win676Ka2P3edENUnEvc/O+YrqDpa8D+kRLmJn9jiP37ghRaCG+Q/Wl9thOXz8j+gI65YIaixP/iiF/OdcdM/CjaWhncZ8j/cUV67XUXlP2h/10g2Lci/lV5UIAnw6z/rwiDjnpPvPwNY/7TduPY/C9WivEGv8D8G9LWrij0BQKCXBMvk7ABAMJyqmSGBsr8soYhviBIEQKCD7jEpXOe/rI2hKcsO9j8KV9n5+uzkv3xtNWjHVuI/gcj/c1e3A0Di2AqgelL1PwUjm8zb9wlA0MowWaKm8T/uu+X/F+AGQIQFUEld0v4/0NASCAIwuL/OFmwBzuzxPzD9Ox5GALs/Nfwz7ijABkDGAvBsiHP9P/Ee96YgofQ/pLk8PShJ+z/kepaVPeHEP06O+8x54tU/1JOdTxHd3T93FcxUvFbuP5qxHSuKdfI/LJjI0GrU0r+aL39EA8z7P8K9u3Ytyt0/8J9uEIqvtD/qwc1TCacBQOYpgmpGwwNA5LmpeT46+b/ont1Zcmfkvw==","dtype":"float64","shape":[50]},"y_values":{"__ndarray__":"ilX+dDpTFEBENUo7fJUCQNGpqOvtIwZA3oTo5mZ3EEABeUQshQEQQHeUJfgt1xFAKhm4FigfFUAgz/EyRWwRQGu4cqTKjQ9Ao2CKD1G6EkB+gMFul3ARQPCspEiG0RRAlsR39YfbEkAjMIpCz1sYQF6rf7/CjBJAdFPkN89sEkCIHBvP1X4QQPg8zbA7jQhA0k2jr/URB0AEkkycsxcXQLPEIhesBBlASSpeVqrzDED57eOd9xATQLgda6rcGhZAJeNQ/3fMGEBMyBfxmwoUQPoGXJSdywZA46j2H7fSEkBSr84l8qUHQN61IqQWnBBAE9f6N9ZuFUB+s3AZQQgXQEnuBh6szA1A4EA5/LCcCkAKmAvWM34MQIQCQ+f3mAtAqxoPHyi7EUBFSx3Mup4EQPEEvDVDhw9AM4lt8/rsDkD0XrCnsO0SQE9GeSBPNw9AC+BKjr0XFUDOoJNmrLMRQKlah8M4/AhAWoGNMUTDEkAJ6jCwUAYWQKrOawT/ZBBALTffN2LUE0CrE+lvO10QQA==","dtype":"float64","shape":[50]}}},"id":"db458b9d-de16-4834-8745-63fca72d846c","type":"ColumnDataSource"},{"attributes":{},"id":"1ddd1958-f264-459b-a1d2-67e43f5c0667","type":"BasicTickFormatter"},{"attributes":{"axis_label":"B","formatter":{"id":"5fbad35a-cfab-4cb6-a890-baa6a2bfd512","type":"BasicTickFormatter"},"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"},"ticker":{"id":"3bdaa512-79cf-4615-8dc1-d03882a5862a","type":"BasicTicker"}},"id":"307b6ef7-b212-4224-924b-398287865847","type":"LinearAxis"},{"attributes":{"items":[{"id":"5e8f5500-fd97-4f2e-8f8b-4227487ebeee","type":"LegendItem"},{"id":"8dd4dce9-d4fa-4be9-89a4-d185dc8c9d28","type":"LegendItem"}],"location":"top_left","plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"}},"id":"3b105053-e882-43c5-a277-ca58530ff314","type":"Legend"},{"attributes":{},"id":"5fbad35a-cfab-4cb6-a890-baa6a2bfd512","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"db458b9d-de16-4834-8745-63fca72d846c","type":"ColumnDataSource"},"glyph":{"id":"5d2f81df-555b-4bca-a5d4-6d1a75b4289e","type":"Circle"},"hover_glyph":null,"muted_glyph":null},"id":"a4089889-af08-46b0-8bf3-95e74b9cb240","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.7},"fill_color":{"value":"#f22c40"},"line_color":{"value":"#f22c40"},"size":{"units":"screen","value":8},"x":{"field":"x_values"},"y":{"field":"y_values"}},"id":"5d2f81df-555b-4bca-a5d4-6d1a75b4289e","type":"Circle"},{"attributes":{"below":[{"id":"8388cfe7-6c9f-4419-b393-ecdc67067023","type":"LinearAxis"}],"css_classes":null,"left":[{"id":"307b6ef7-b212-4224-924b-398287865847","type":"LinearAxis"}],"renderers":[{"id":"ca65b176-6fa4-4c09-8bca-cf8713f07aaf","type":"BoxAnnotation"},{"id":"d5f68dac-36f9-431e-a4cd-46015ad258dc","type":"GlyphRenderer"},{"id":"a4089889-af08-46b0-8bf3-95e74b9cb240","type":"GlyphRenderer"},{"id":"3b105053-e882-43c5-a277-ca58530ff314","type":"Legend"},{"id":"8388cfe7-6c9f-4419-b393-ecdc67067023","type":"LinearAxis"},{"id":"307b6ef7-b212-4224-924b-398287865847","type":"LinearAxis"},{"id":"03ef4186-ba8a-4ba3-a6b0-bc3f842fadc3","type":"Grid"},{"id":"f4f53c53-b8f7-40de-9f63-0dd744ecb57e","type":"Grid"}],"title":{"id":"631b788a-3ef7-4cee-af7f-35a669cea9b2","type":"Title"},"tool_events":{"id":"c7cd1701-0cb3-4406-8a79-58b86cd15e05","type":"ToolEvents"},"toolbar":{"id":"062e2493-37b6-4e74-9f56-a84f07e2e047","type":"Toolbar"},"x_mapper_type":"auto","x_range":{"id":"b340b6ab-809b-46bc-b6d9-0f40e31cc9f9","type":"Range1d"},"y_mapper_type":"auto","y_range":{"id":"11e03262-a815-4021-9435-7abc1aebf5ca","type":"Range1d"}},"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"},{"attributes":{"fill_alpha":{"value":0.7},"fill_color":{"value":"#5ab738"},"line_color":{"value":"#5ab738"},"size":{"units":"screen","value":8},"x":{"field":"x_values"},"y":{"field":"y_values"}},"id":"0753fb25-20c7-4241-92bf-91da2ebd5afe","type":"Circle"},{"attributes":{"data_source":{"id":"67fbe450-3226-406e-ac4f-906e983c3bdc","type":"ColumnDataSource"},"glyph":{"id":"0753fb25-20c7-4241-92bf-91da2ebd5afe","type":"Circle"},"hover_glyph":null,"muted_glyph":null},"id":"d5f68dac-36f9-431e-a4cd-46015ad258dc","type":"GlyphRenderer"},{"attributes":{"callback":null,"end":7.012592944176847,"start":-2.083769713783157},"id":"11e03262-a815-4021-9435-7abc1aebf5ca","type":"Range1d"},{"attributes":{"callback":null,"end":5.03992549965526,"start":-2.178232940646024},"id":"b340b6ab-809b-46bc-b6d9-0f40e31cc9f9","type":"Range1d"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"ca65b176-6fa4-4c09-8bca-cf8713f07aaf","type":"BoxAnnotation"},{"attributes":{"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"}},"id":"08a31718-e7dd-4f5b-a505-5113f7891039","type":"PanTool"},{"attributes":{},"id":"78e017e3-27cb-4851-8311-4f48daf8415b","type":"BasicTicker"},{"attributes":{"axis_label":"A","formatter":{"id":"1ddd1958-f264-459b-a1d2-67e43f5c0667","type":"BasicTickFormatter"},"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"},"ticker":{"id":"78e017e3-27cb-4851-8311-4f48daf8415b","type":"BasicTicker"}},"id":"8388cfe7-6c9f-4419-b393-ecdc67067023","type":"LinearAxis"},{"attributes":{"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"}},"id":"d3ae0191-1f23-42d5-8310-28d53b19ed96","type":"HelpTool"},{"attributes":{"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"}},"id":"08a87d2a-d5ad-4790-a44c-a0911f8a9159","type":"ResetTool"},{"attributes":{"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"}},"id":"4f81e6ac-ed95-4a21-904e-02c87228424c","type":"SaveTool"},{"attributes":{"overlay":{"id":"ca65b176-6fa4-4c09-8bca-cf8713f07aaf","type":"BoxAnnotation"},"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"}},"id":"ad3ab7b4-8f94-4fb7-8836-9c5e251b1f41","type":"BoxZoomTool"},{"attributes":{},"id":"3bdaa512-79cf-4615-8dc1-d03882a5862a","type":"BasicTicker"},{"attributes":{"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"}},"id":"95334ee8-1ec8-4c18-a637-0597637f7b3b","type":"WheelZoomTool"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"a4089889-af08-46b0-8bf3-95e74b9cb240","type":"GlyphRenderer"}]},"id":"8dd4dce9-d4fa-4be9-89a4-d185dc8c9d28","type":"LegendItem"},{"attributes":{"dimension":1,"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"},"ticker":{"id":"3bdaa512-79cf-4615-8dc1-d03882a5862a","type":"BasicTicker"}},"id":"f4f53c53-b8f7-40de-9f63-0dd744ecb57e","type":"Grid"},{"attributes":{"plot":null,"text":null},"id":"631b788a-3ef7-4cee-af7f-35a669cea9b2","type":"Title"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"08a31718-e7dd-4f5b-a505-5113f7891039","type":"PanTool"},{"id":"95334ee8-1ec8-4c18-a637-0597637f7b3b","type":"WheelZoomTool"},{"id":"ad3ab7b4-8f94-4fb7-8836-9c5e251b1f41","type":"BoxZoomTool"},{"id":"4f81e6ac-ed95-4a21-904e-02c87228424c","type":"SaveTool"},{"id":"08a87d2a-d5ad-4790-a44c-a0911f8a9159","type":"ResetTool"},{"id":"d3ae0191-1f23-42d5-8310-28d53b19ed96","type":"HelpTool"}]},"id":"062e2493-37b6-4e74-9f56-a84f07e2e047","type":"Toolbar"},{"attributes":{},"id":"c7cd1701-0cb3-4406-8a79-58b86cd15e05","type":"ToolEvents"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"d5f68dac-36f9-431e-a4cd-46015ad258dc","type":"GlyphRenderer"}]},"id":"5e8f5500-fd97-4f2e-8f8b-4227487ebeee","type":"LegendItem"},{"attributes":{"callback":null,"column_names":["x_values","y_values"],"data":{"C":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"chart_index":[{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1},{"C":1}],"x_values":{"__ndarray__":"7BuzX7/fEEAEa7k2sAztP8VLr9xF8glAGGEHh1/TD0DysMIanAf0PwiGTfs0+vo/nhk6S6KY/D9SCSsn78ARQGRA6MqJdAlAHvkLidao5z9+2MqtzJD0PzILJDAgDfU/IqKmxdugBEDalA0q3Pf0P2nATcv60wJAcTm2fBt0A0BsxP+ppG3uPxi1Z6o1nANAfj07oPzr+T9W/g4nTxj2P2rU36/byv8/+BfQjF4M4j+sUz5MhjrsP8qPBgOJxgFA7gaMlrwgAEApS1Stajj7P1KCBz4dT/8/lEpt8Kpl6D9AaQO/g5j5P5mnS1QbzwdAsmv9calxBUBYh1JC87X2P5PO/SsM0gVAeT1QUm/q+D8sObFtSj0JQL55m0IxRPo/psC72fJhD0Ai+3f5HVbwP4L1DovSWgVA/szUnMsa8z/u9R3wKAUIQOz6KgJtxfU/XfaMvgsOBUATR2+f0J0GQPFZKxv6SwdAdNn7t3RH8j+Jb3FjfuHwP9gG3WIdlvY/PsnHCJoJCEBaSgvY+43qPw==","dtype":"float64","shape":[50]},"y_values":{"__ndarray__":"MhW62Z/fAUDE8yHqE27dP6iscvwGb/M/2pGFbIIGA0CkguvGeaPKPySxZp6qEOU/RHFqFrEz+z+BuakXa3n9P5jEDIiNVMe/3GMOILjo2z8WvnNUmP32PxIcNlKChgNAiJF8aFo85z9Y7os+zzuyP6wZc37aygFA+NxBtYXSyb+pIHU9/2TuP1DsIvXLAsm/RekNEbX5BUAmX6Goar7tP6SEVFBX4wRAweWUKZRk9T8qV5OM87oGQFH+OUvrhfg/0JBO87hO0b+MQiVu4Yvev3DYWkzJ+c0/a23Hs6ui8j+WsNOX2UjtPz9lv7xWdvM/jq7cRmWK5r+1Y99JgC/5P4s8tEXL4fQ/CsBPyRieBkCcMT9kytj4P/i+7iULOsM/dv3jQP3b/D8MTDY5pET5P4xbTFowHv0/PkxIazp2BkC6trtp0cPnPwXRpcPa7+c/SI7n/EYP5j/sKvpC54n7P5Q4FA/Wkc8/ZbJFhRAeAEB4McenGrXkv4otJp06NvW/1N6WZuGG7z9qbH1SV9/7Pw==","dtype":"float64","shape":[50]}}},"id":"67fbe450-3226-406e-ac4f-906e983c3bdc","type":"ColumnDataSource"},{"attributes":{"plot":{"id":"cb444657-1944-4bb1-a42a-014651bd7ba6","subtype":"Chart","type":"Plot"},"ticker":{"id":"78e017e3-27cb-4851-8311-4f48daf8415b","type":"BasicTicker"}},"id":"03ef4186-ba8a-4ba3-a6b0-bc3f842fadc3","type":"Grid"}],"root_ids":["cb444657-1944-4bb1-a42a-014651bd7ba6"]},"title":"Bokeh Application","version":"0.12.5"}};
            var render_items = [{"docid":"1ff71593-dc0c-44f0-85a1-ded984c09026","elementid":"f4f7048c-d2e2-4b58-86d4-06a2a4f056a0","modelid":"cb444657-1944-4bb1-a42a-014651bd7ba6"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("f4f7048c-d2e2-4b58-86d4-06a2a4f056a0")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>


# Logistic regression


```python
from sklearn.linear_model import LogisticRegression
```


```python
classifier = LogisticRegression()
```


```python
x_train.shape
```




    (75, 4)




```python
y_train.shape
```




    (75,)




```python
classifier.fit(x_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
pred = classifier.predict(x_test)
```


```python
pred
```




    array([0, 2, 1, 0, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1,
           1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 2, 2, 0, 1, 0, 2, 2, 2,
           0, 2, 1, 1, 0, 1, 2, 1, 1, 0, 0, 1, 1, 0, 2, 1, 0, 2, 0, 2, 1, 1, 2,
           0, 2, 1, 0, 0, 1])



Accuracy = What fraction of population was corerctly predicted?


```python
np.mean(pred == y_test)
```




    0.92000000000000004




```python
classifier.score(x_test, y_test)
```




    0.92000000000000004




```python
classifier.score(x_train, y_train)
```




    0.95999999999999996



# KNN


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier(n_neighbors=1)
```


```python
knn.fit(x_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=1, p=2,
               weights='uniform')




```python
plt.scatter(x[y==0,0], x[y==0,1],
           c='blue', s=40, label = 'o')
plt.scatter(x[y==1,0], x[y==1,1],
           c='red', s=40, label='1', marker = 's')
```




    <matplotlib.collections.PathCollection at 0x14caea03c50>




![png](output_61_1.png)



```python
knn.score(x_test, y_test)
```




    0.93333333333333335




```python
from sklearn.datasets import load_iris
iris = load_iris()
```


```python
x = iris.data
y = iris.target
```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   train_size = 0.5, test_size = 0.5, random_state = 123, stratify = y)
print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(y_train) / float(len(y_train)) * 100.0)
print('Test:', np.bincount(y_test) / float(len(y_test)) * 100.0)
```

    All: [ 33.33333333  33.33333333  33.33333333]
    Training: [ 33.33333333  33.33333333  33.33333333]
    Test: [ 33.33333333  33.33333333  33.33333333]
    


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 102)

scores = []
k_values = np.arange(1,10)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))

plt.plot(k_values, scores)
plt.xlabel('# Neighbors')
plt.ylabel('Accuracy')
```




    <matplotlib.text.Text at 0x14caeb7acc0>




![png](output_66_1.png)


# Supervised learning
## Linear regression


```python
x = np.linspace(-3,3,100)
print(x)
```

    [-3.         -2.93939394 -2.87878788 -2.81818182 -2.75757576 -2.6969697
     -2.63636364 -2.57575758 -2.51515152 -2.45454545 -2.39393939 -2.33333333
     -2.27272727 -2.21212121 -2.15151515 -2.09090909 -2.03030303 -1.96969697
     -1.90909091 -1.84848485 -1.78787879 -1.72727273 -1.66666667 -1.60606061
     -1.54545455 -1.48484848 -1.42424242 -1.36363636 -1.3030303  -1.24242424
     -1.18181818 -1.12121212 -1.06060606 -1.         -0.93939394 -0.87878788
     -0.81818182 -0.75757576 -0.6969697  -0.63636364 -0.57575758 -0.51515152
     -0.45454545 -0.39393939 -0.33333333 -0.27272727 -0.21212121 -0.15151515
     -0.09090909 -0.03030303  0.03030303  0.09090909  0.15151515  0.21212121
      0.27272727  0.33333333  0.39393939  0.45454545  0.51515152  0.57575758
      0.63636364  0.6969697   0.75757576  0.81818182  0.87878788  0.93939394
      1.          1.06060606  1.12121212  1.18181818  1.24242424  1.3030303
      1.36363636  1.42424242  1.48484848  1.54545455  1.60606061  1.66666667
      1.72727273  1.78787879  1.84848485  1.90909091  1.96969697  2.03030303
      2.09090909  2.15151515  2.21212121  2.27272727  2.33333333  2.39393939
      2.45454545  2.51515152  2.57575758  2.63636364  2.6969697   2.75757576
      2.81818182  2.87878788  2.93939394  3.        ]
    


```python
rng = np.random.RandomState(42)
```


```python
y = np.sin(4*x) + x + 0.1*rng.uniform(size=len(x))
```


```python
plt.plot(x, y, 'o')
```




    [<matplotlib.lines.Line2D at 0x14caf091ac8>]




![png](output_71_1.png)



```python
print('Before: ', x.shape)
X = x[:, np.newaxis]
print('After: ', X.shape)
```

    Before:  (100,)
    After:  (100, 1)
    


```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
```


```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print('Weight coeffs: ', regressor.coef_)
print('y-axis intercept: ', regressor.intercept_)
```

    Weight coeffs:  [ 0.9151048]
    y-axis intercept:  0.0390406342807
    


```python
min_pt = X.min() * regressor.coef_[0] + regressor.intercept_
max_pt = X.max() * regressor.coef_[0] + regressor.intercept_

plt.plot([X.min(), X.max()], [min_pt, max_pt])
plt.plot(x_train, y_train, 'o');
```


![png](output_76_0.png)



```python
y_pred_train = regressor.predict(x_train)
```


```python
plt.plot(x_train, y_train, 'o', label="data")
plt.plot(x_train, y_pred_train, 'o', label="prediction")
plt.plot([X.min(),X.max()], [min_pt, max_pt], label = 'fit')
plt.legend(loc='best')
```




    <matplotlib.legend.Legend at 0x14caeec3e48>




![png](output_78_1.png)


For regression scikit learn by default takes the R2 score.


```python
regressor.score(x_test, y_test)
```




    0.81833419325851742


