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

import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import React from 'react'
import { Link } from 'react-router-dom'

import { useStyles } from './Style'
import Title from './Title'

class SessionList extends React.Component {
  constructor(props) {
    super(props)
    this.state = {}
  }

  refreshInfo() {
    fetch('api/session')
      .then((res) => res.json())
      .then((res) => {
        this.setState(res)
      })
  }

  componentDidMount() {
    if (this.interval !== undefined) {
      clearInterval(this.interval)
    }
    this.interval = setInterval(() => this.refreshInfo(), 5000)
    this.refreshInfo()
  }

  componentWillUnmount() {
    clearInterval(this.interval)
  }

  render() {
    if (this.state === undefined || this.state.sessions === undefined) {
      return <div>Loading</div>
    }

    return (
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell style={{ fontWeight: 'bolder' }}>Session ID</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {this.state.sessions.map((session) => (
            <TableRow key={`session_row_${session.session_id}`}>
              <TableCell>
                <Link to={`/session/${session.session_id}/task`}>
                  {session.session_id}
                </Link>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    )
  }
}

export default function SessionListPage() {
  const classes = useStyles()
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Title>Sessions</Title>
      </Grid>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <SessionList />
        </Paper>
      </Grid>
    </Grid>
  )
}
