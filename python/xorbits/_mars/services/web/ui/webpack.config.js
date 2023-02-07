const path = require('path');

module.exports = {
  entry: './src/App.js',
  mode: "production",
  resolve: {
    extensions: ['*', '.js', '.jsx'],
  },
  output: {
    path: path.resolve(__dirname, '../static'),
    filename: 'bundle.js',
    publicPath: 'static/',
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel-loader',
      }
    ]
  }
};
