import React from 'react';
import Title from '../Title';
import Paper from '@material-ui/core/Paper';
import {formatTime} from '../Utils';
import Button from '@material-ui/core/Button';
import SaveIcon from '@material-ui/icons/Save';
import Grid from '@material-ui/core/Grid';
import PropTypes from 'prop-types';


export default class NodeLogTab extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loaded: false,
      content: null,
      interval: null,
      endpoint: null
    };
  }

  componentDidMount() {
    const intervalId = setInterval(() => this.loadLogs(), 10000);
    // store intervalId in the state, so it can be accessed later
    this.setState({
      interval: intervalId
    });
    this.loadLogs();
  }

  componentWillUnmount() {
    if (this.state.interval != null) {
      clearInterval(this.state.interval);
    }
  }

  render() {
    if (!this.state.loaded) {
      return (
        <div>Loading</div>
      );
    }
    return (
      <div>
        <Grid container spacing={2}>
          <Grid item xs={8}>
            <Title component="h3">Generate Time: {formatTime(this.getTimestamp() / 1000)}</Title>
          </Grid>
          <Grid item xs>
            <Button
              variant="contained"
              color="primary"
              size="small"
              startIcon={<SaveIcon/>}
              onClick={() => this.downloadLogs()}
            >Save
            </Button>
          </Grid>
        </Grid>
        <div>
          <Paper style={{width: '100%', overflow: 'auto'}}>
            <pre style={{fontSize: 'smaller'}}>{this.state.content}</pre>
          </Paper>
        </div>
      </div>
    );
  }

  loadLogs() {
    fetch(`api/cluster/logs?address=${this.props.endpoint}`)
      .then((res) => res.json())
      .then((res) => {
        this.setState({
          loaded: true,
          content: res.content
        });
      });
  }

  getTimestamp() {
    return new Date().getTime();
  }

  downloadLogs() {
    const filename = ''.concat(
      'xorbits_',
      this.props.role, '_',
      this.props.endpoint, '_',
      this.getTimestamp().toString(), '_',
      'log.txt');
    fetch(`api/cluster/logs?address=${this.props.endpoint}&&size=-1`)
      .then(res => res.blob().then(blob => {
        let a = document.createElement('a');
        let url = window.URL.createObjectURL(blob);
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
        a = null;
      }));
  }
}

NodeLogTab.propTypes = {
  endpoint: PropTypes.string,
  role: PropTypes.string
};
