// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tasks

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/apache/arrow/go/v17/arrow/memory"
)

var defaultAllocator = memory.DefaultAllocator

const (
	groupByColumnPrefix = "$group_by_"
	elementIndicesCol   = "$element_indices"
)

func groupByColumnName(fieldID int64) string {
	return fmt.Sprintf("%s%d", groupByColumnPrefix, fieldID)
}

func isGroupByColumnName(name string) bool {
	if !strings.HasPrefix(name, groupByColumnPrefix) {
		return false
	}
	fieldID := strings.TrimPrefix(name, groupByColumnPrefix)
	if fieldID == "" {
		return false
	}
	_, err := strconv.ParseInt(fieldID, 10, 64)
	return err == nil
}
