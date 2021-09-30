# Class_perceptron
#  The Command Used  -


```bash

git add . && git commit -m "docstring updated" && git push origin main

```


# Add Img -
![sample Image](plots/and.png)


## python code

``` python

def main(data, eta, epochs, filename , PlotFileName):
    df = pd.DataFrame(data)
    print(df)
    X,y = prepare_data(df)
    
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model , filename= filename)
    save_plot(df, PlotFileName, model)

```