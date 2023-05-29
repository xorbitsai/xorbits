/*
 * Copyright 2022-2023 XProbe Inc.
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

import Paper from '@mui/material/Paper'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Grid from '@mui/material/Unstable_Grid2'
import sum from 'lodash/sum'
import PropTypes from 'prop-types'
import React from 'react'

import { StyledTableCell, StyledTableRow, useStyles } from './Style'
import Title from './Title'
import { toReadableSize } from './Utils'

class NodeInfo extends React.Component {
  constructor(props) {
    super(props)
    this.nodeRole = props.nodeRole.toLowerCase()
    this.state = {
      version: {},
    }
  }

  refreshInfo() {
    const roleId = this.nodeRole === 'supervisor' ? 0 : 1
    fetch(
      `api/cluster/nodes?role=${roleId}&env=1&resource=1&exclude_statuses=-1`
    )
      .then((res) => res.json())
      .then((res) => {
        const { state } = this
        state[this.nodeRole] = res.nodes
        this.setState(state)
      })

    if (JSON.stringify(this.state.version) === '{}') {
      fetch(`/api/xorbits/version`)
        .then((res) => res.json())
        .then((res) => {
          const { state } = this
          state['version'] = {
            release: 'v' + res['version'],
            commit: res['full-revisionid'],
          }
          this.setState(state)
        })
    }
  }

  componentDidMount() {
    this.interval = setInterval(() => this.refreshInfo(), 5000)
    this.refreshInfo()
  }

  componentWillUnmount() {
    clearInterval(this.interval)
  }

  render() {
    if (this.state === undefined || this.state[this.nodeRole] === undefined) {
      return <div>Loading</div>
    }

    const roleData = this.state[this.nodeRole]

    const gatherResourceStats = (prop) =>
      sum(
        Object.values(roleData).map((val) =>
          sum(Object.values(val.resource).map((a) => a[prop]))
        )
      )

    const resourceStats = {
      cpu_total: gatherResourceStats('cpu_total'),
      cpu_avail: gatherResourceStats('cpu_avail'),
      memory_total: gatherResourceStats('memory_total'),
      memory_avail: gatherResourceStats('memory_avail'),
      gpu_total: gatherResourceStats('gpu_total'),
      gpu_avail: gatherResourceStats('gpu_avail'),
      gpu_memory_total: gatherResourceStats('gpu_memory_total'),
      gpu_memory_avail: gatherResourceStats('gpu_memory_avail'),
    }

    resourceStats.cpu_used = resourceStats.cpu_total - resourceStats.cpu_avail
    resourceStats.memory_used =
      resourceStats.memory_total - resourceStats.memory_avail
    resourceStats.gpu_used = resourceStats.gpu_total - resourceStats.gpu_avail
    resourceStats.gpu_memory_used =
      resourceStats.gpu_memory_total - resourceStats.gpu_memory_avail

    return (
      <Table size="small">
        <TableHead>
          <TableRow>
            <StyledTableCell style={{ fontWeight: 'bolder' }}>
              Item
            </StyledTableCell>
            <StyledTableCell style={{ fontWeight: 'bolder' }}>
              <Grid container>
                <Grid>Value</Grid>
              </Grid>
            </StyledTableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <StyledTableRow>
            <StyledTableCell>Count</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid>{Object.keys(this.state[this.nodeRole]).length}</Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
          <StyledTableRow>
            <StyledTableCell>CPU Info</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={4}>
                  Usage:
                  {resourceStats.cpu_used.toFixed(2)}
                </Grid>
                <Grid xs={8}>
                  Total:
                  {resourceStats.cpu_total.toFixed(2)}
                </Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
          <StyledTableRow>
            <StyledTableCell>CPU Memory Info</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={4}>
                  Usage: {toReadableSize(resourceStats.memory_used)}
                </Grid>
                <Grid xs={8}>
                  Total: {toReadableSize(resourceStats.memory_total)}
                </Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
          <StyledTableRow>
            <StyledTableCell>GPU Info</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={4}>
                  Usage:
                  {resourceStats.gpu_used.toFixed(2)}
                </Grid>
                <Grid xs={8}>
                  Total:
                  {resourceStats.gpu_total.toFixed(2)}
                </Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
          <StyledTableRow>
            <StyledTableCell>GPU Memory Info</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={4}>
                  Usage: {toReadableSize(resourceStats.gpu_memory_used)}
                </Grid>
                <Grid xs={8}>
                  Total: {toReadableSize(resourceStats.gpu_memory_total)}
                </Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
          <StyledTableRow>
            <StyledTableCell>Version</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={4}>Release: {this.state.version.release}</Grid>
                <Grid xs={8}>Commit: {this.state.version.commit}</Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
        </TableBody>
      </Table>
    )
  }
}

NodeInfo.propTypes = {
  nodeRole: PropTypes.string,
}

export default function Dashboard() {
  const classes = useStyles()
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <Title>Supervisors</Title>
          <NodeInfo nodeRole="supervisor" />
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <Title>Workers</Title>
          <NodeInfo nodeRole="worker" />
        </Paper>
      </Grid>
    </Grid>
  )
}
