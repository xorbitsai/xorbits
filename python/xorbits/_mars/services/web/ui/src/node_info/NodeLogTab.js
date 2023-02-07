/*
 * Copyright 2022 XProbe Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import React from 'react';
import Title from '../Title';
import Paper from '@material-ui/core/Paper';
import {formatTime} from '../Utils';
import Button from '@material-ui/core/Button';
import SaveIcon from '@material-ui/icons/Save';
import Grid from '@material-ui/core/Grid';
import PropTypes from 'prop-types';
import streamSaver from 'streamsaver';


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
      'mars_',
      this.props.role, '_',
      this.props.endpoint, '_',
      this.getTimestamp().toString(), '_',
      'log.txt');
    fetch(`api/cluster/logs?address=${this.props.endpoint}&&size=-1`)
      .then(response => {
        const fileStream = streamSaver.createWriteStream(filename);
        const readableStream = response.body;
        // more optimized pipe version
        // (Safari may have pipeTo, but it's useless without the WritableStream)
        if (window.WritableStream && readableStream.pipeTo) {
          return readableStream.pipeTo(fileStream);
        }
        // Write (pipe) manually
        window.writer = fileStream.getWriter();
        let writer = window.writer;

        const reader = readableStream.getReader();
        const pump = () => reader.read()
          .then(res => res.done
            ? writer.close()
            : writer.write(res.value).then(pump));
        pump();
      });
  }
}

NodeLogTab.propTypes = {
  endpoint: PropTypes.string,
  role: PropTypes.string
};
