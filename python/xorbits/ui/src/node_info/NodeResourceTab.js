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

import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Grid from '@mui/material/Unstable_Grid2';
import PropTypes from 'prop-types';
import React, {Fragment} from 'react';

import {StyledPaper} from '../Style';
import Title from '../Title';
import { toReadableSize } from '../Utils';

export default class NodeResourceTab extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loaded: false,
    };
  }

  refreshInfo() {
    fetch(`api/cluster/nodes?nodes=${this.props.endpoint
    }&resource=1&detail=1&exclude_statuses=-1`)
      .then((res) => res.json())
      .then((res) => {
        const result = res.nodes[this.props.endpoint];
        const { state } = this;
        state.loaded = true;
        state.resource = result.resource;
        state.detail = result.detail;
        this.setState(state);
      });
  }

  componentDidMount() {
    this.interval = setInterval(() => this.refreshInfo(), 5000);
    this.refreshInfo();
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  generateBandRows() {
    const converted = {};
    Object.keys(this.state.resource).map((band) => {
      let detail = this.state.resource[band];
      converted[band] = {
        'CPU': {},
        'Memory': {}
      };
      let cpuAvail = detail['cpu_avail'];
      let cpuTotal = detail['cpu_total'];
      let cpuUsage = (cpuTotal - cpuAvail).toFixed(2);

      let memAvail = detail['memory_avail'];
      let memTotal = detail['memory_total'];
      let memUsage = toReadableSize(memTotal - memAvail);

      converted[band]['CPU']['CpuUsage'] = cpuUsage;
      converted[band]['CPU']['CpuTotal'] = cpuTotal.toFixed(2);

      converted[band]['Memory']['MemUsage'] = memUsage;
      converted[band]['Memory']['MemTotal'] = toReadableSize(memTotal);
    });

    return (
      <Grid item xs={12}>
        <StyledPaper>
          <Title>Bands</Title>
          <Table sx={{ minWidth: 650 }} aria-label="simple table">
            <TableHead>
              <TableRow>
                <TableCell style={{ fontWeight: 'bolder' }}>Band</TableCell>
                <TableCell align="right" style={{ fontWeight: 'bolder' }}>Item</TableCell>
                <TableCell align="right" style={{ fontWeight: 'bolder' }}>Usage</TableCell>
                <TableCell align="right" style={{ fontWeight: 'bolder' }}>Total</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.keys(converted).map((band) => (
                <Fragment key={band}>
                  <TableRow>
                    <TableCell rowSpan={3}>
                      {band}
                    </TableCell>
                  </TableRow>

                  {Object.keys(converted[band]).map((item) => (
                    <TableRow key={item}>
                      <TableCell align="right">
                        {item}
                      </TableCell>
                      {Object.keys(converted[band][item]).map((k) => (
                        <TableCell key={k} align="right">
                          {converted[band][item][k]}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </Fragment>))}
            </TableBody>
          </Table>
        </StyledPaper>
      </Grid>
    );
  }

  generateIOSummaryRows() {
    return (
      <Grid item xs={12}>
        <StyledPaper>
          <Title>IO Summary</Title>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell style={{ fontWeight: 'bolder' }}>Item</TableCell>
                <TableCell align="right" style={{ fontWeight: 'bolder' }}>Read</TableCell>
                <TableCell align="right" style={{ fontWeight: 'bolder' }}>Write</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                <TableCell style={{ width: '20%' }}>Disk</TableCell>
                <TableCell align="right" style={{ width: '40%' }} >{this.state.detail.disk.reads.toFixed(2)}</TableCell>
                <TableCell align="right" style={{ width: '40%' }} >{this.state.detail.disk.writes.toFixed(2)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell style={{ width: '20%' }}>Network</TableCell>
                <TableCell align="right" style={{ width: '40%' }}>{this.state.detail.network.receives.toFixed(2)}</TableCell>
                <TableCell align="right" style={{ width: '40%' }}>{this.state.detail.network.sends.toFixed(2)}</TableCell>
              </TableRow>
              {Boolean(this.state.detail.iowait) &&
                  <Fragment>
                    <TableRow>
                      <TableCell rowSpan={2}/>
                      <TableCell align="right" style={{fontWeight: 'bolder'}}>Info</TableCell>
                      <TableCell align="right" style={{fontWeight: 'bolder'}}>Value</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell align="right">IO Wait</TableCell>
                      <TableCell align="right">{this.state.detail.iowait.toFixed(2)}</TableCell>
                    </TableRow>
                  </Fragment>
              }
            </TableBody>
          </Table>
        </StyledPaper>
      </Grid>
    );
  }

  generateDiskRows() {
    return (
      <Grid item xs={12}>
        <StyledPaper>
          <Title>Disks</Title>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell/>
                <TableCell align="center" style={{fontWeight: 'bolder'}} colSpan={2}>Size</TableCell>
                <TableCell align="center" style={{fontWeight: 'bolder'}} colSpan={2}>INode</TableCell>
              </TableRow>
              <TableRow>
                <TableCell style={{fontWeight: 'bolder'}}>Path</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Usage</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Total</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Usage</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Total</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {
                Object.keys(this.state.detail.disk.partitions).map((path) => (
                  <TableRow key={path}>
                    <TableCell>{path}</TableCell>
                    <TableCell align="right">{toReadableSize(this.state.detail.disk.partitions[path].size_used)}</TableCell>
                    <TableCell align="right">{toReadableSize(this.state.detail.disk.partitions[path].size_total)}</TableCell>
                    <TableCell align="right">{toReadableSize(this.state.detail.disk.partitions[path].inode_used)}</TableCell>
                    <TableCell align="right">{toReadableSize(this.state.detail.disk.partitions[path].inode_total)}</TableCell>
                  </TableRow>
                ))
              }
            </TableBody>
          </Table>
        </StyledPaper>
      </Grid>
    );
  }

  generateQuotaRows() {
    return (
      <Grid item xs={12}>
        <StyledPaper>
          <Title>Quota</Title>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell style={{fontWeight: 'bolder'}}>Band</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Hold</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Allocated</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Total</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.keys(this.state.detail.quota).map((band) => (
                <TableRow key={`${band}-quota`}>
                  <TableCell>{band}</TableCell>
                  <TableCell align="right">{toReadableSize(this.state.detail.quota[band].hold_size)}</TableCell>
                  <TableCell align="right">{toReadableSize(this.state.detail.quota[band].allocated_size)}</TableCell>
                  <TableCell align="right">{toReadableSize(this.state.detail.quota[band].quota_size)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </StyledPaper>
      </Grid>
    );
  }

  generateSlotsRows() {
    return (
      <Grid item xs={6}>
        <StyledPaper>
          <Title>Slot</Title>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell style={{fontWeight: 'bolder'}}>Band</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Slots</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.keys(this.state.detail.slot).map((band) => (
                <TableRow key={`${band}-slot`}>
                  <TableCell>{band}</TableCell>
                  <TableCell align="right">{this.state.detail.slot[band].length}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </StyledPaper>
      </Grid>
    );
  }

  generateStorageRows() {
    return (
      <Grid item xs={6}>
        <StyledPaper>
          <Title>Storage</Title>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell style={{fontWeight: 'bolder'}}>Band</TableCell>
                <TableCell style={{fontWeight: 'bolder'}}>Level</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Used</TableCell>
                <TableCell align="right" style={{fontWeight: 'bolder'}}>Total</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.keys(this.state.detail.storage).map((band) => (
                <Fragment key={band}>
                  <TableRow>
                    <TableCell rowSpan={Object.keys(this.state.detail.storage[band]).length + 1}>
                      {band}
                    </TableCell>
                  </TableRow>
                  {Object.keys(this.state.detail.storage[band]).map((level) => (
                    <TableRow key={level}>
                      <TableCell>{level}</TableCell>
                      <TableCell align="right">{toReadableSize(this.state.detail.storage[band][level].size_used)}</TableCell>
                      <TableCell align="right">{toReadableSize(this.state.detail.storage[band][level].size_total)}</TableCell>
                    </TableRow>
                  ))}
                </Fragment>
              ))}
            </TableBody>
          </Table>
        </StyledPaper>
      </Grid>
    );
  }

  render() {
    if (!this.state.loaded) {
      return (
        <div>Loading</div>
      );
    }
    return (
      <Grid container spacing={2}>
        {this.generateBandRows()}
        {this.generateIOSummaryRows()}
        {Boolean(this.state.detail.disk.partitions) &&
            this.generateDiskRows()
        }
        {Object.keys(this.state.detail.quota).length > 0 &&
            this.generateQuotaRows()
        }
        {Object.keys(this.state.detail.slot).length > 0 &&
            this.generateSlotsRows()
        }
        {Object.keys(this.state.detail.storage).length > 0 &&
            this.generateStorageRows()
        }
      </Grid>
    );
  }
}

NodeResourceTab.propTypes = {
  endpoint: PropTypes.string,
};
